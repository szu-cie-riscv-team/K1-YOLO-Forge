"""One implementation shared by calibration and FP32/INT8 evaluation."""
from __future__ import annotations

import cv2
import numpy as np


def letterbox(bgr, hw):
    height, width = bgr.shape[:2]
    out_h, out_w = map(int, hw)
    ratio = min(out_w / width, out_h / height)
    new_w, new_h = round(width * ratio), round(height * ratio)
    left = round((out_w - new_w) / 2 - 0.1)
    top = round((out_h - new_h) / 2 - 0.1)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(resized, top, out_h-new_h-top, left, out_w-new_w-left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / np.float32(255)
    return tensor, {"ratio": ratio, "left": left, "top": top, "source_hw": [height, width]}


def read_image(path):
    # imdecode handles non-ASCII Linux paths as well as ASCII paths.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def xslim_preprocess(path_list, input_parametr):
    import torch
    hw = input_parametr["input_shape"][-2:]
    return torch.from_numpy(np.concatenate([letterbox(read_image(p), hw)[0] for p in path_list], axis=0))
