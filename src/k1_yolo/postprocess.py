"""NumPy nine-output DFL decoding shared by host and K1 inference."""
from __future__ import annotations
import numpy as np


def box_iou(a,b):
    if not len(a) or not len(b):
        return np.zeros((len(a),len(b)),dtype=np.float32)
    lt=np.maximum(a[:,None,:2],b[None,:,:2]);rb=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.maximum(rb-lt,0);inter=wh[:,:,0]*wh[:,:,1]
    area_a=np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0)
    area_b=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    return inter/(area_a[:,None]+area_b[None,:]-inter+1e-9)


def decode(outputs,hw,meta,nc,confidence=.001,nms_iou=.7,max_detections=300):
    candidates=[]
    for branch,stride in enumerate([8,16,32]):
        box,score,summary=outputs[branch*3:branch*3+3]
        h,w=hw[0]//stride,hw[1]//stride
        if list(box.shape)!=[1,64,h,w] or list(score.shape)!=[1,nc,h,w] or not all(np.isfinite(v).all() for v in [box,score,summary]):
            raise ValueError("Invalid nine-output inference result")
        logits=box[0].reshape(4,16,-1);exp=np.exp(logits-logits.max(axis=1,keepdims=True))
        distances=(exp/exp.sum(axis=1,keepdims=True)*np.arange(16)[None,:,None]).sum(axis=1)
        probabilities=score[0].reshape(nc,-1)
        if probabilities.min() < -1e-5 or probabilities.max() > 1+1e-5:
            raise ValueError("Expected class probabilities, not logits")
        classes=probabilities.argmax(axis=0);conf=probabilities.max(axis=0)
        yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing="ij")
        centers=np.stack([xx.ravel()+.5,yy.ravel()+.5])
        xyxy=np.concatenate([centers-distances[:2],centers+distances[2:]],axis=0).T*stride
        keep=conf>=confidence
        candidates.append(np.column_stack([xyxy[keep],conf[keep],classes[keep]]))
    predictions=np.concatenate(candidates)
    if not len(predictions):
        return np.empty((0,6),dtype=np.float32)
    # Bound worst-case NMS work independently of output count.
    order=np.argsort(-predictions[:,4],kind="stable")[:30000]
    keep=[]
    while len(order) and len(keep)<max_detections:
        i=order[0];keep.append(i);order=order[1:]
        overlaps=box_iou(predictions[i:i+1,:4],predictions[order,:4])[0]
        order=order[(overlaps<=nms_iou)|(predictions[order,5]!=predictions[i,5])]
    result=predictions[keep].copy()
    result[:,[0,2]]=(result[:,[0,2]]-meta["left"])/meta["ratio"]
    result[:,[1,3]]=(result[:,[1,3]]-meta["top"])/meta["ratio"]
    sh,sw=meta["source_hw"]
    result[:,[0,2]]=result[:,[0,2]].clip(0,sw)
    result[:,[1,3]]=result[:,[1,3]].clip(0,sh)
    return result
