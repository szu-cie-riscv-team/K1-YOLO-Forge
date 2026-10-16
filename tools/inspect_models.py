"""Read-only ONNX graph inventory; a checker result is not a K1 runtime test."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def tensor_info(value):
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim
        ],
    }


def inspect_model(path: Path):
    model = onnx.load(path)
    initializers = {value.name: value for value in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}

    def constant(name):
        if name in initializers:
            return initializers[name]
        node = producers.get(name)
        if node and node.op_type == "DequantizeLinear":
            return initializers.get(node.input[0])
        return None

    convs, special_nodes = [], []
    for node in model.graph.node:
        if node.op_type == "Conv":
            weight = constant(node.input[1])
            convs.append({
                "name": node.name,
                "weight_shape": list(weight.dims) if weight is not None else None,
                "attributes": {a.name: json_value(helper.get_attribute_value(a)) for a in node.attribute},
            })
        if node.op_type in {"Resize", "Clip", "ReduceSum", "LeakyRelu"}:
            special_nodes.append({
                "name": node.name,
                "op_type": node.op_type,
                "attributes": {a.name: json_value(helper.get_attribute_value(a)) for a in node.attribute},
                "constant_inputs": {
                    name: json_value(numpy_helper.to_array(initializers[name]))
                    for name in node.input if name in initializers
                },
            })

    checker = {"status": "PASS"}
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        checker = {"status": "FAIL", "error": str(exc)}
    metadata_keys = Counter(item.key for item in model.metadata_props)
    duplicate_keys = sorted(key for key, count in metadata_keys.items() if count > 1)
    diagnostic = None
    if duplicate_keys:
        # Diagnose metadata independently. Never rewrite the supplied model.
        copy = onnx.ModelProto()
        copy.CopyFrom(model)
        metadata = {item.key: item.value for item in copy.metadata_props}
        del copy.metadata_props[:]
        for key, value in metadata.items():
            item = copy.metadata_props.add()
            item.key, item.value = key, value
        try:
            onnx.checker.check_model(copy)
            diagnostic = {"status": "PASS", "scope": "in-memory copy with metadata keys deduplicated; source unchanged"}
        except Exception as exc:
            diagnostic = {"status": "FAIL", "error": str(exc)}

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ir_version": model.ir_version,
        "opset_import": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
        "producer": {"name": model.producer_name, "version": model.producer_version},
        "inputs": [tensor_info(value) for value in model.graph.input],
        "outputs": [tensor_info(value) for value in model.graph.output],
        "operator_counts": dict(sorted(Counter(
            f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node
        ).items())),
        "initializer_dtype_counts": dict(Counter(
            onnx.TensorProto.DataType.Name(value.data_type) for value in initializers.values()
        )),
        "functions": [{"domain": f.domain, "name": f.name} for f in model.functions],
        "metadata": [{"key": item.key, "value": item.value} for item in model.metadata_props],
        "duplicate_metadata_keys": duplicate_keys,
        "checker": checker,
        "metadata_only_diagnostic": diagnostic,
        "convolutions": convs,
        "special_nodes": special_nodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Prevent an accidental overwrite of any model supplied as input.
    if args.output.resolve() in {path.resolve() for path in args.models}:
        parser.error("Output must differ from every input model")
    report = {
        "schema_version": 1,
        "scope": "static graph inspection; no inference or hardware benchmark",
        "environment": {
            "python": sys.version.split()[0],
            **{name: importlib.metadata.version(name) for name in ["onnx", "numpy", "protobuf"]},
        },
        "models": [inspect_model(path) for path in args.models],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model in report["models"]:
        print(f"{model['file']}: checker={model['checker']['status']} sha256={model['sha256']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
