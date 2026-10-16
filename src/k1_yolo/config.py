from __future__ import annotations

import copy
from pathlib import Path
import yaml
from .classes import class_map


def merge(base, update, prefix=""):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict) and key not in {"custom_slots", "expected_counts"}:
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping: {prefix}{key}")
            result[key] = merge(base[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


class Config:
    def __init__(self, filename, root=None):
        self.root = Path(root or Path.cwd()).resolve()
        self.filename = Path(filename).resolve()
        defaults = yaml.safe_load((self.root / "config.example.yaml").read_text(encoding="utf-8"))
        supplied = yaml.safe_load(self.filename.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be a YAML mapping")
        self.raw = merge(defaults, supplied)
        self.classes = class_map(self.raw["classes"])
        self.validate()

    def __getitem__(self, key):
        return self.raw[key]

    def path(self, value):
        if value is None:
            return None
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def hw(self, stage):
        return tuple(self[stage]["input_hw"] or self["model"]["input_hw"])

    def validate(self):
        c = self.raw
        if c["schema_version"] != 1:
            raise ValueError("Unsupported config schema")
        import re
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", c["project"]["name"]):
            raise ValueError("project.name must be a short alphanumeric/hyphen/underscore name")
        for stage in ["train", "export"]:
            size = self.hw(stage)
            if len(size) != 2 or any(type(x) is not int or x <= 0 or x % 32 for x in size):
                raise ValueError(f"{stage} input_hw must be [height,width], positive multiples of 32")
        if c["model"]["architecture"] != "yolov8n_standard":
            raise ValueError("Only yolov8n_standard is implemented; the 43-Conv reference is not trainable here yet")
        if c["export"]["opset"] != 17 or c["quantize"]["opset"] != 17:
            raise ValueError("This validated K1 route requires opset=17 at export and XSlim stages")
        if c["export"]["batch"] != 1 or c["export"]["dynamic"] or c["export"]["embed_nms"]:
            raise ValueError("K1 export requires static batch=1 without embedded NMS")
        if c["export"]["reg_max"] != 16 or c["export"]["output_format"] != "yolov8_dfl_9outputs":
            raise ValueError("Only nine-output YOLOv8 reg_max=16 is implemented")
        if c["export"]["allowed_custom_domains"]:
            raise ValueError("Custom ONNX domains are not accepted on this K1 route")
        p = c["preprocess"]
        expected = dict(color="rgb", layout="nchw", dtype="float32", resize="letterbox", interpolation="linear", padding_value=114)
        if any(p[k] != v for k, v in expected.items()) or abs(p["scale"] - 1 / 255) > 1e-12:
            raise ValueError("Preprocessing must match the implemented RGB/letterbox114/float32/255 route")
        split = c["data"]["split"]
        if split["mode"] not in {"provided", "counts", "ratios"} or split["count_policy"] != "exact":
            raise ValueError("Invalid split mode or count policy")
        for name in ["ratios", "counts", "expected_counts"]:
            if set(split[name]) != {"train", "val", "test"}:
                raise ValueError(f"split.{name} must contain train, val, test")
        if split["mode"] == "ratios" and (any(v < 0 for v in split["ratios"].values()) or abs(sum(split["ratios"].values()) - 1) > 1e-8):
            raise ValueError("Split ratios must be nonnegative and sum to 1")
        if any(type(v) is not int or v < 0 for v in split["counts"].values()):
            raise ValueError("Split counts must be nonnegative integers")
        if any(v is not None and (type(v) is not int or v < 0) for v in split["expected_counts"].values()):
            raise ValueError("expected_counts must be null or nonnegative integers")
        if c["data"]["split_by"] not in {"capture_group", "image"}:
            raise ValueError("split_by must be capture_group or image")
        if c["data"]["calibration_source"] not in {"train", "directory"}:
            raise ValueError("calibration_source must be train or directory")
        if self.classes["mode"] == "replace_coco":
            r = c["classes"]["replace_coco"]["replay_train_fraction"]
            if not 0 < r < 1:
                raise ValueError("replay_train_fraction must lie strictly between 0 and 1")
        for field in ["epochs", "patience"]:
            if type(c["train"][field]) is not int or c["train"][field] < (1 if field == "epochs" else 0):
                raise ValueError(f"Invalid train.{field}")
        if c["train"]["batch"] != "auto" and (type(c["train"]["batch"]) is not int or c["train"]["batch"] < 1):
            raise ValueError("train.batch must be auto or a positive integer")
        if c["train"]["amp"] not in {"auto", True, False}:
            raise ValueError("train.amp must be auto or boolean")
        q = c["quantize"]
        if q["tool"] != "xslim" or str(q["candidate_version"]) != "2.1.2":
            raise ValueError("This pipeline implements official XSlim 2.1.2")
        if q["precision_level"] not in {0, 1, 2} or q["finetune_level"] not in {0, 1, 2, 3}:
            raise ValueError("Invalid static INT8 XSlim levels")
        if q["calibration"]["batch_size"] != 1 or q["calibration"]["preprocess"] != "shared_letterbox":
            raise ValueError("Static model calibration uses batch=1 and shared_letterbox")
        if type(q["calibration"]["max_samples"]) is not int or q["calibration"]["max_samples"] < 1:
            raise ValueError("calibration.max_samples must be positive")
        if not 10<=q["calibration"]["max_samples"]<=1000:
            raise ValueError("XSlim 2.1.2 calibration.max_samples must lie in 10..1000")
        if q["calibration"]["type"] not in {"default","kl","minmax","percentile","mse"}:
            raise ValueError("Unknown XSlim calibration type")
        if type(c["project"]["seed"]) is not int:
            raise ValueError("project.seed must be an integer")
        for section in ["evaluation","deploy"]:
            for field in ["confidence_threshold","nms_iou_threshold"]:
                if not isinstance(c[section][field],(int,float)) or not 0<c[section][field]<1:
                    raise ValueError(f"{section}.{field} must be between 0 and 1")
            if type(c[section]["max_detections"]) is not int or c[section]["max_detections"]<1:
                raise ValueError(f"{section}.max_detections must be positive")
        drop=c["evaluation"]["max_map_drop"]
        if drop is not None and (not isinstance(drop,(int,float)) or not 0<=drop<=1):
            raise ValueError("evaluation.max_map_drop must be null or an absolute AP drop in 0..1")
