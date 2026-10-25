from __future__ import annotations

from pathlib import Path
import shutil
import uuid
import yaml

from .common import digest,read_json,write_json
from .exporting import check_onnx


def package(cfg,run):
    report=read_json(run/"evaluation/val.json")
    if report["accuracy_gate"]=="FAIL":
        raise ValueError("Cannot package a model that failed the configured accuracy gate")
    final_target=cfg.path("dist")/cfg["project"]["name"]/run.name
    target=final_target.with_name(final_target.name+".building-"+uuid.uuid4().hex[:8])
    target.mkdir(parents=True,exist_ok=False)
    model=run/"quantize/model.q.onnx"
    contract=check_onnx(model,cfg.hw("export"),cfg.classes["nc"],True,cfg["model"]["architecture"])
    if digest(model)!=report["models"]["int8"]["model_sha256"]:
        raise ValueError("Quantized model changed since evaluation")
    shutil.copy2(model,target/"model.q.onnx")
    shutil.copy2(run/"prepare/class_map.json",target/"class_map.json")
    shutil.copy2(run/"evaluation/val.json",target/"evaluation.json")
    (target/"labels.txt").write_text("\n".join(cfg.classes["names"])+"\n",encoding="utf-8")
    inference={"model_path":"model.q.onnx","labels_path":"labels.txt","output_format":"yolov8_dfl_9outputs",
               "num_classes":cfg.classes["nc"],"reg_max":16,"input_height":cfg.hw("export")[0],"input_width":cfg.hw("export")[1],
               "preprocess":cfg["preprocess"],"deploy":cfg["deploy"]}
    (target/"inference.yaml").write_text(yaml.safe_dump(inference,allow_unicode=True),encoding="utf-8")
    write_json(target/"inference.json",inference)
    runtime=target/"runtime/k1_yolo";runtime.mkdir(parents=True)
    for name in ["__init__.py","preprocess.py","postprocess.py"]:
        shutil.copy2(Path(__file__).with_name(name),runtime/name)
    shutil.copy2(cfg.root/"tools/run_k1.py",target/"run_k1.py")
    manifest={"schema_version":1,"run_id":run.name,"scheme":cfg["workflow"]["scheme"],
              "architecture":cfg["model"]["architecture"],"class_mode":cfg.classes["mode"],
              "names":cfg.classes["names"],"model_contract":contract,"dataset_manifest_sha256":digest(run/"prepare/manifest.json"),
              "training":read_json(run/"training.json") if (run/"training.json").exists() else {"checkpoint":"externally supplied"},
              "quantization":read_json(run/"quantize/contract.json"),"evaluation":report,
              "board_validation":"NOT_RUN","deployment_status":"HOST_GRAPH_VALIDATED_BOARD_UNVERIFIED"}
    (target/"README.md").write_text(
        "# K1 YOLO deployment package\n\n"
        "Host graph and numerical checks completed; K1 hardware acceptance is NOT_RUN.\n\n"
        "1. Use the matching official SpacemiT ONNX Runtime and SpaceMITExecutionProvider on K1.\n"
        "2. Resolve model/label paths relative to this directory; follow inference.yaml.\n"
        "3. The model has FLOAT32 RGB NCHW inputs and nine FLOAT32 outputs; internal weights/activations are quantized.\n"
        "4. Decode DFL externally (reg_max=16), then class-aware NMS and inverse letterbox.\n"
        "5. Existing k1muse_perception fixes classes at 80. For N != 80, adapt its engine validation and decoder before integration; this package does not modify ROS.\n"
        "   Standalone N-class inference is already included: python3 run_k1.py --image IMAGE --profile.\n"
        "6. Verify EP subgraph assignment and real images before recording K1 latency and memory.\n\n"
        "Evaluation is COCO-style AP on YOLO TXT boxes without crowd/ignore annotations. Host CPU timing is not K1 timing.\n"
        "Check manifest.json for accuracy gate and independent-test status.\n",encoding="utf-8")
    manifest["files"]={p.relative_to(target).as_posix():digest(p) for p in target.rglob("*") if p.is_file()}
    write_json(target/"manifest.json",manifest)
    if final_target.exists():
        shutil.move(final_target,final_target.with_name(final_target.name+".previous-"+uuid.uuid4().hex[:8]))
    target.rename(final_target);target=final_target
    archive=Path(shutil.make_archive(str(target),"zip",root_dir=target.parent,base_dir=target.name))
    write_json(run/"package.json",{"directory":str(target),"archive":str(archive),"sha256":digest(archive),"board_validation":"NOT_RUN"})
    return run/"package.json"
