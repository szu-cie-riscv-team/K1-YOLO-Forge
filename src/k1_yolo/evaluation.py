from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from .common import digest, environment, read_json, write_json
from .data import verify_manifest
from .exporting import check_onnx
from .preprocess import letterbox, read_image


from .postprocess import box_iou,decode


def match_predictions(pred,gt,thresholds):
    correct=np.zeros((len(pred),len(thresholds)),dtype=bool)
    if not len(gt):
        return correct
    overlaps=box_iou(pred[:,:4],gt[:,1:])
    for ti,threshold in enumerate(thresholds):
        used=set()
        for pi in np.argsort(-pred[:,4],kind="stable"):
            available=[gi for gi in range(len(gt)) if gi not in used and gt[gi,0]==pred[pi,5] and overlaps[pi,gi]>=threshold]
            if available:
                gi=max(available,key=lambda index:overlaps[pi,index]);used.add(gi);correct[pi,ti]=True
    return correct


def metrics(predictions,ground_truth,names,operating_confidence):
    thresholds=np.arange(.5,.951,.05)
    stats=[];targets=[]
    for pred,gt in zip(predictions,ground_truth):
        stats.append((match_predictions(pred,gt,thresholds),pred[:,4],pred[:,5]))
        targets.extend(gt[:,0].astype(int).tolist())
    tp=np.concatenate([s[0] for s in stats]);conf=np.concatenate([s[1] for s in stats]);cls=np.concatenate([s[2] for s in stats])
    per_class={}
    for ci,name in enumerate(names):
        n=sum(t==ci for t in targets)
        selected=np.flatnonzero(cls==ci);selected=selected[np.argsort(-conf[selected],kind="stable")]
        ap=[]
        if n:
            for ti in range(len(thresholds)):
                positives=tp[selected,ti].astype(float)
                recall=np.cumsum(positives)/n
                precision=np.cumsum(positives)/np.arange(1,len(selected)+1)
                # COCO-style 101 recall points; no crowds/ignore regions in YOLO TXT.
                ap.append(float(np.mean([np.max(precision[recall>=r],initial=0.) for r in np.linspace(0,1,101)])))
        fixed=selected[conf[selected]>=operating_confidence]
        good=int(tp[fixed,0].sum())
        per_class[name]={"targets":n,"predictions_at_threshold":len(fixed),"precision":good/len(fixed) if len(fixed) else 0.,
                         "recall":good/n if n else None,"ap50":ap[0] if ap else None,"ap50_95":float(np.mean(ap)) if ap else None}
    available=[v for v in per_class.values() if v["targets"]]
    return {"ap50":float(np.mean([v["ap50"] for v in available])) if available else None,
            "ap50_95":float(np.mean([v["ap50_95"] for v in available])) if available else None,
            "per_class":per_class,"protocol":"101 recall points; IoU .50:.05:.95; best-class anchors; class-aware NMS; no crowd annotations",
            "operating_confidence":operating_confidence}


def evaluate(cfg,run,subset="val"):
    import onnxruntime as ort
    manifest=verify_manifest(run)
    rows=[r for r in manifest["images"] if r["split"]==subset]
    if not rows:
        raise ValueError(f"No {subset} images; do not substitute validation for independent test")
    settings=cfg["evaluation"]
    reports={}
    folder=run/"evaluation";folder.mkdir(parents=True,exist_ok=True)
    for name,path in [("fp32",run/"export/model.fp32.onnx"),("int8",run/"quantize/model.q.onnx")]:
        check_onnx(path,cfg.hw("export"),cfg.classes["nc"],quantized=name=="int8")
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
        session.run(None,{"images":np.zeros((1,3,*cfg.hw("export")),dtype=np.float32)})
        predictions=[];truth=[];timings=[];previews=[]
        for row in rows:
            tensor,meta=letterbox(read_image(row["prepared_image"]),cfg.hw("export"))
            start=time.perf_counter();outputs=session.run(None,{"images":tensor});timings.append((time.perf_counter()-start)*1000)
            pred=decode(outputs,cfg.hw("export"),meta,cfg.classes["nc"],settings["confidence_threshold"],settings["nms_iou_threshold"],settings["max_detections"])
            gt=np.asarray(row["labels"],dtype=float).reshape(-1,5)
            if len(gt):
                sh,sw=row["shape_hw"]
                cls=gt[:,0:1];centers=gt[:,1:3]*[sw,sh];half=gt[:,3:5]*[sw,sh]/2
                gt=np.concatenate([cls,centers-half,centers+half],axis=1)
            truth.append(gt);predictions.append(pred)
            if len(previews)<8:
                import cv2
                image=read_image(row["prepared_image"])
                for x1,y1,x2,y2,score,cls in pred[pred[:,4]>=cfg["deploy"]["confidence_threshold"]]:
                    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    cv2.putText(image,f"{cfg.classes['names'][int(cls)]} {score:.2f}",(int(x1),max(15,int(y1))),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                cv2.imwrite(str(folder/f"{subset}-{name}-{len(previews):02d}.jpg"),image);previews.append(row["sha256"])
        result=metrics(predictions,truth,cfg.classes["names"],cfg["deploy"]["confidence_threshold"])
        result.update(model_sha256=digest(path),bytes=path.stat().st_size,images=len(rows),
                      host_cpu_inference_ms={"p50":float(np.percentile(timings,50)),"p95":float(np.percentile(timings,95))})
        reports[name]=result
    drop=None if reports["fp32"]["ap50_95"] is None else reports["fp32"]["ap50_95"]-reports["int8"]["ap50_95"]
    max_drop=settings["max_map_drop"]
    report={"split":subset,"models":reports,"map50_95_drop":drop,
            "accuracy_gate":"NOT_APPLICABLE" if drop is None else ("NOT_CONFIGURED" if max_drop is None else ("PASS" if drop<=max_drop else "FAIL")),
            "maximum_accepted_drop":max_drop,"independent_test_present":any(r["split"]=="test" for r in manifest["images"]),
            "environment":environment(),"board_validation":"NOT_RUN"}
    write_json(folder/f"{subset}.json",report)
    print(f"{subset}: FP32 mAP50-95={reports['fp32']['ap50_95']}, INT8={reports['int8']['ap50_95']}, drop={drop}")
    if report["accuracy_gate"]=="FAIL":
        raise ValueError(f"Quantization exceeds max_map_drop={max_drop}; report saved")
    return folder/f"{subset}.json"
