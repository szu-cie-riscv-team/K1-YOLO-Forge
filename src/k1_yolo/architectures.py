"""Architecture registration for the K1-oriented YOLO reference approximation."""
from __future__ import annotations

import torch.nn as nn
from ultralytics.nn.modules.conv import RepConv as UltralyticsRepConv


class K1RepConv(UltralyticsRepConv):
    """RepVGG-style block with the activation used by the reference ONNX."""

    default_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)


def register_k1_fast_modules():
    """Teach the pinned Ultralytics 8.3.200 parser about LeakyReLU RepConv."""
    import inspect
    import ultralytics.nn.tasks as tasks
    if tasks.RepConv is K1RepConv:
        return K1RepConv

    # Upstream 8.3.200's parser does not include RepConv in base_modules. Extend
    # only that parser set so YAML RepConv layers get the normal inferred c1/c2.
    # Pinning ultralytics is intentional; fail loudly if its parser changes.
    source = inspect.getsource(tasks.parse_model)
    needle = "            Classify,\n            Conv,\n            ConvTranspose,"
    replacement = "            Classify,\n            Conv,\n            K1RepConv,\n            ConvTranspose,"
    if source.count(needle) != 1:
        raise RuntimeError("Ultralytics parse_model changed; update K1RepConv parser integration")
    namespace = tasks.__dict__.copy()
    namespace["K1RepConv"] = K1RepConv
    namespace["RepConv"] = K1RepConv
    exec(compile(source.replace(needle, replacement), inspect.getsourcefile(tasks.parse_model), "exec"), namespace)
    tasks.parse_model = namespace["parse_model"]
    # DetectionModel.fuse resolves this symbol in tasks globals and recognizes
    # K1RepConv as a reparameterizable block because it is the subclass here.
    tasks.RepConv = K1RepConv
    return K1RepConv


def fast_model_yaml() -> str:
    from pathlib import Path
    return str(Path(__file__).with_name("models") / "yolov8n_k1_fast.yaml")
