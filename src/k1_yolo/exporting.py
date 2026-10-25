from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .common import digest, environment, read_json, write_json
from .data import verify_manifest
from .preprocess import letterbox, read_image


def expected_shapes(hw, nc):
    h,w = hw
    return [[1,c,h//s,w//s] for s in [8,16,32] for c in [64,nc,1]]


def check_onnx(path, hw, nc, quantized=False, architecture="yolov8n_standard"):
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)
    versions = {v.domain: v.version for v in model.opset_import}
    if versions.get("", versions.get("ai.onnx")) != 17:
        raise ValueError(f"Expected opset 17, got {versions}")
    if any(n.domain not in {"", "ai.onnx"} for n in model.graph.node) or model.functions:
        raise ValueError("Unexpected custom domain/function in K1 nine-output route")
    if model.ir_version > 9:
        raise ValueError(f"IR {model.ir_version} exceeds this pipeline's reference range (8/9)")
    def shape(value):
        return [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]
    if len(model.graph.input) != 1 or shape(model.graph.input[0]) != [1,3,*hw]:
        raise ValueError("Unexpected model input shape")
    if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for value in [*model.graph.input,*model.graph.output]):
        raise ValueError("Model I/O must remain float32")
    if [shape(v) for v in model.graph.output] != expected_shapes(hw,nc):
        raise ValueError(f"Expected nine 64/{nc}/1 outputs at strides 8/16/32")
    counts = {}
    initializers={v.name:v for v in model.graph.initializer}
    producers={name:node for node in model.graph.node for name in node.output}
    def const_initializer(name):
        seen=set()
        while name in producers and name not in seen:
            seen.add(name)
            node=producers[name]
            if node.op_type in {"Identity","DequantizeLinear"} and node.input:
                name=node.input[0]
            else:
                return None
        return initializers.get(name)
    conv_weight_parameters=0
    conv_bias_parameters=0
    allowed={"Conv","Sigmoid","Mul","Split","Add","Concat","MaxPool","Resize","ReduceSum","Clip",
             "LeakyRelu","QuantizeLinear","DequantizeLinear","Constant","Identity"}
    for node in model.graph.node:
        if node.op_type not in allowed:
            raise ValueError(f"Operator outside the inspected K1 reference route: {node.op_type}")
        counts[node.op_type] = counts.get(node.op_type,0)+1
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Conv" and "kernel_shape" not in attrs:
            raise ValueError(f"Conv missing kernel_shape: {node.name}")
        if node.op_type=="Conv":
            weight=const_initializer(node.input[1])
            if weight is None:
                raise ValueError(f"Conv weight must be constant or constant QDQ: {node.name}")
            conv_weight_parameters+=int(np.prod(weight.dims))
            if len(node.input)>2:
                bias=const_initializer(node.input[2])
                if bias is not None:
                    conv_bias_parameters+=int(np.prod(bias.dims))
        if node.op_type == "Resize" and (attrs.get("mode",b"nearest") not in {b"nearest",b"linear"} or
                attrs.get("coordinate_transformation_mode",b"half_pixel") not in {b"asymmetric",b"half_pixel"}):
            raise ValueError(f"Unsupported Resize attributes: {node.name}")
    qdq = counts.get("QuantizeLinear",0)>0 and counts.get("DequantizeLinear",0)>0
    if qdq != quantized:
        raise ValueError(f"Expected quantized={quantized}, got QDQ={qdq}")
    if architecture == "yolov8n_k1_fast":
        reference_counts = {"Conv":43,"LeakyRelu":37,"Sigmoid":3,"MaxPool":3,"Resize":2,
                            "Concat":5,"ReduceSum":3,"Clip":3}
        mismatches = {name:(counts.get(name,0),expected) for name,expected in reference_counts.items()
                      if counts.get(name,0) != expected}
        if mismatches:
            raise ValueError(f"K1-fast graph differs from the inspected deployment profile: {mismatches}")
        if counts.get("Mul",0) or counts.get("Split",0):
            raise ValueError("K1-fast graph unexpectedly contains standard YOLOv8 SiLU/C2f operators")
        if nc == 80 and (conv_weight_parameters,conv_bias_parameters)!=(1_815_472,3_264):
            raise ValueError("80-class K1-fast Conv parameter profile differs from the inspected reference")
    return {"opsets":versions,"ir_version":model.ir_version,"operators":counts,"shapes":expected_shapes(hw,nc),
            "architecture":architecture,"conv_weight_parameters":conv_weight_parameters,
            "conv_bias_parameters":conv_bias_parameters,"sha256":digest(path),"static_check":"PASS","board_validation":"NOT_RUN"}


def raw_model(checkpoint, names, architecture="yolov8n_standard"):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
    if architecture == "yolov8n_k1_fast":
        from .architectures import register_k1_fast_modules
        register_k1_fast_modules()
    model = YOLO(str(checkpoint)).model.cpu().float().eval()
    actual = list(model.names.values()) if isinstance(model.names,dict) else model.names
    if actual != names:
        raise ValueError(f"Checkpoint classes differ from configured names: {actual}")
    model.fuse(verbose=False)
    from ultralytics.nn.modules.block import C2f
    for module in model.modules():
        if isinstance(module,C2f):
            # Fixed channel splits avoid Shape/Gather/chunk bookkeeping in the
            # static K1 graph, as in Ultralytics' static export path.
            module.forward=module.forward_split
    head = model.model[-1]
    if not isinstance(head,Detect) or head.reg_max != 16 or head.nl != 3 or head.end2end:
        raise ValueError("Expected standard three-scale YOLOv8 Detect head")

    class RawHead(torch.nn.Module):
        def __init__(self, source, reference_score_clip=False):
            super().__init__()
            self.cv2,self.cv3 = source.cv2,source.cv3
            self.reference_score_clip = reference_score_clip
            for key in ["f","i","type","np"]:
                if hasattr(source,key):
                    setattr(self,key,getattr(source,key))

        def forward(self,features):
            outputs = []
            for i in range(3):
                box = self.cv2[i](features[i])
                score = self.cv3[i](features[i]).sigmoid()
                summary = score.sum(1, keepdim=True)
                if self.reference_score_clip:
                    summary = summary.clamp(0, 1)
                outputs.extend([box,score,summary])
            return tuple(outputs)
    model.model[-1] = RawHead(head, architecture == "yolov8n_k1_fast")
    return model


def export(cfg,run,checkpoint=None):
    import torch
    import onnx
    import onnxruntime as ort
    manifest = verify_manifest(run)
    checkpoint = Path(checkpoint) if checkpoint else Path(read_json(run/"training.json")["best"])
    model = raw_model(checkpoint,cfg.classes["names"],cfg["model"]["architecture"])
    hw = cfg.hw("export")
    folder = run/"export";folder.mkdir(parents=True,exist_ok=True)
    output = folder/"model.fp32.onnx"
    names = [f"s{s}_{kind}" for s in [8,16,32] for kind in ["box","score","sum"]]
    torch.onnx.export(model,torch.zeros(1,3,*hw),str(output),opset_version=17,
                      input_names=["images"],output_names=names,dynamic_axes=None,
                      do_constant_folding=True,dynamo=False)
    graph = onnx.load(output)
    metadata = {"names":json.dumps(cfg.classes["names"],ensure_ascii=False),"input_hw":json.dumps(hw),
                "output_format":"yolov8_dfl_9outputs","reg_max":"16","license":"AGPL-3.0; see THIRD_PARTY.md"}
    del graph.metadata_props[:]
    for k,v in metadata.items():
        prop = graph.metadata_props.add();prop.key,prop.value=k,v
    onnx.save(graph,output)
    contract = check_onnx(output,hw,cfg.classes["nc"],architecture=cfg["model"]["architecture"])
    session = ort.InferenceSession(str(output),providers=["CPUExecutionProvider"])
    comparisons = []
    samples = [r for r in manifest["images"] if r["split"]=="val"][:3]
    for row in samples:
        tensor = letterbox(read_image(row["path"]),hw)[0]
        with torch.inference_mode():
            reference = [value.numpy() for value in model(torch.from_numpy(tensor))]
        exported = session.run(None,{"images":tensor})
        errors=[]
        for a,b in zip(reference,exported):
            if not np.allclose(a,b,rtol=2e-4,atol=2e-4):
                raise ValueError(f"PyTorch/ONNX mismatch: {row['path']}; max error={np.max(np.abs(a-b))}")
            errors.append(float(np.max(np.abs(a-b))))
        comparisons.append({"image_sha256":row["sha256"],"max_abs_error_per_output":errors})
    contract.update(scheme=cfg["workflow"]["scheme"],checkpoint_sha256=digest(checkpoint),
                    numerical_comparison=comparisons,environment=environment())
    write_json(folder/"contract.json",contract)
    return output
