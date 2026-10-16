#!/usr/bin/env python3
"""Standalone model-package inference. Default requires the official SpaceMIT EP."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--cpu",action="store_true",help="Explicit host CPU smoke check; does not validate K1")
    parser.add_argument("--repeat",type=int,default=20)
    parser.add_argument("--report",type=Path,default=Path("inference-result.json"))
    parser.add_argument("--output-image",type=Path)
    parser.add_argument("--profile",action="store_true")
    args=parser.parse_args()
    if args.repeat<1:parser.error("--repeat must be positive")
    root=args.package.resolve()
    sys.path.insert(0,str(root/"runtime"))
    from k1_yolo.preprocess import letterbox,read_image
    from k1_yolo.postprocess import decode
    manifest=json.loads((root/"manifest.json").read_text())
    settings=json.loads((root/"inference.json").read_text())
    model=root/"model.q.onnx"
    actual_hash=hashlib.sha256(model.read_bytes()).hexdigest()
    if actual_hash!=manifest["model_contract"]["sha256"]:
        raise ValueError("Package model SHA256 differs from manifest")
    if not args.cpu:
        import spacemit_ort  # Registers the official provider plugin.
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=settings["deploy"]["ort_threads"]["intra"]
    options.inter_op_num_threads=settings["deploy"]["ort_threads"]["inter"]
    provider_options={"SPACEMIT_EP_INTRA_THREAD_NUM":str(settings["deploy"]["ep_threads"]["intra"]),
                      "SPACEMIT_EP_INTER_THREAD_NUM":str(settings["deploy"]["ep_threads"]["inter"])}
    if args.profile:
        options.enable_profiling=True
        provider_options.update(SPACEMIT_EP_DUMP_SUBGRAPHS="1",SPACEMIT_EP_DEBUG_PROFILE="k1-ep-profile")
    provider="CPUExecutionProvider" if args.cpu else "SpaceMITExecutionProvider"
    session=ort.InferenceSession(str(model),sess_options=options,providers=[provider],provider_options=[provider_options])
    if provider not in session.get_providers():
        raise RuntimeError(f"Requested {provider} did not initialize; refusing fallback")
    image=read_image(args.image)
    hw=[settings["input_height"],settings["input_width"]]
    tensor,meta=letterbox(image,hw)
    session.run(None,{session.get_inputs()[0].name:tensor})
    timings=[]
    for _ in range(args.repeat):
        start=time.perf_counter();outputs=session.run(None,{session.get_inputs()[0].name:tensor})
        timings.append((time.perf_counter()-start)*1000)
    deploy=settings["deploy"]
    predictions=decode(outputs,hw,meta,len(manifest["names"]),deploy["confidence_threshold"],deploy["nms_iou_threshold"],deploy["max_detections"])
    detections=[]
    for x1,y1,x2,y2,score,cls in predictions:
        detections.append({"class_id":int(cls),"label":manifest["names"][int(cls)],"score":float(score),"xyxy":[float(v) for v in [x1,y1,x2,y2]]})
        cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),(0,255,0),2)
    result={"provider":provider,"providers":session.get_providers(),"model_sha256":actual_hash,"detections":detections,
            "inference_ms":{"p50":float(np.percentile(timings,50)),"p95":float(np.percentile(timings,95))},
            "scope":"HOST_CPU_SMOKE" if args.cpu else "K1_IMAGE_SMOKE",
            "network_ep_assignment":"NOT_MEASURED; inspect profile/subgraph dumps separately",
            "full_board_acceptance":"NOT_RUN"}
    if args.profile:result["ort_profile"]=session.end_profiling()
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if args.output_image:
        args.output_image.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(args.output_image),image)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
