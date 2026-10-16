from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from .common import digest, environment, read_json, write_json, run_logged
from .data import verify_manifest
from .exporting import check_onnx,expected_shapes


def quantize(cfg,run):
    manifest = verify_manifest(run)
    source = run/"export/model.fp32.onnx"
    check_onnx(source,cfg.hw("export"),cfg.classes["nc"])
    if importlib.metadata.version("xslim") != "2.1.2":
        raise ValueError("Quantization requires the separate official XSlim 2.1.2 environment")
    folder = run/"quantize";folder.mkdir(parents=True,exist_ok=True)
    (folder/"calibration.txt").write_text("".join(row["path"]+"\n" for row in manifest["calibration"]),encoding="utf-8")
    count = len(manifest["calibration"])
    if not 10 <= count <= 1000:
        raise ValueError("XSlim calibration_step requires 10..1000 distinct images at batch=1")
    import torch
    from .training import choose_device
    device = cfg["quantize"]["calibration"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    if device not in {"cpu","cuda"} or (device=="cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested calibration device unavailable (use cpu/cuda/auto)")
    config = {
        "model_parameters":{"onnx_model":str(source),"working_dir":str(folder),"output_prefix":"model","opset":17},
        "calibration_parameters":{"calibration_step":count,"calibration_batch_size":1,"calibration_device":device,
            "calibration_type":cfg["quantize"]["calibration"]["type"],"input_parameters":[{
                "input_name":"images","input_shape":[1,3,*cfg.hw("export")],"color_format":"rgb",
                "mean_value":[0.,0.,0.],"std_value":[255.,255.,255.],"file_type":"img",
                "preprocess_file":str(Path(__file__).with_name("preprocess.py"))+":xslim_preprocess",
                "data_list_path":str(folder/"calibration.txt")} ]},
        "quantization_parameters":{"precision_level":cfg["quantize"]["precision_level"],
            "finetune_level":cfg["quantize"]["finetune_level"],"analysis_enable":cfg["quantize"]["analysis_enable"]}}
    write_json(folder/"xslim.json",config)
    target = folder/"model.q.onnx"
    run_logged([sys.executable,"-m","xslim","-c",folder/"xslim.json","-i",source,"-o",target,"--opset","17"],folder/"xslim.log")
    if not target.is_file():
        raise RuntimeError("XSlim did not produce the requested model")
    # XSlim may remove inferred output shapes. Verify runtime shapes before
    # restoring only tensor shape annotations; never alter computation or opset.
    import onnx
    import onnxruntime as ort
    model = onnx.load(target)
    session = ort.InferenceSession(str(target),providers=["CPUExecutionProvider"])
    actual = session.run(None,{"images":np.zeros((1,3,*cfg.hw("export")),dtype=np.float32)})
    expected = expected_shapes(cfg.hw("export"),cfg.classes["nc"])
    if [list(a.shape) for a in actual] != expected:
        raise ValueError("XSlim changed output count or tensor semantics")
    for value,shape in zip(model.graph.output,expected):
        del value.type.tensor_type.shape.dim[:]
        for size in shape:
            value.type.tensor_type.shape.dim.add().dim_value=size
    keys={item.key:item.value for item in model.metadata_props}
    del model.metadata_props[:]
    for key,value in keys.items():
        item=model.metadata_props.add();item.key,item.value=key,value
    onnx.save(model,target)
    contract = check_onnx(target,cfg.hw("export"),cfg.classes["nc"],quantized=True)
    contract.update(source_sha256=digest(source),calibration_images=count,
                    environment=environment(),shape_annotations="restored after actual CPU runtime shape verification")
    write_json(folder/"contract.json",contract)
    return target
