from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os

from .common import digest, environment, read_json, write_json
from .data import verify_manifest


def choose_device(value):
    import torch
    if value == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if str(value) != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"GPU requested ({value}), but CUDA is unavailable")
    # Single GPU is explicit; DDP would need separate class/context propagation.
    if "," in str(value):
        raise ValueError("This version supports one training device per run")
    return str(value)


class FixedInput:
    """Final dataset transform: actual fixed H/W, normalized labels and ratio_pad."""
    def __init__(self, hw):
        self.hw = tuple(hw)

    def __call__(self, sample):
        import torch.nn.functional as F
        image = sample["img"]
        h, w = image.shape[-2:]
        th, tw = self.hw
        if (h, w) == self.hw:
            return sample
        ratio = min(th/h, tw/w)
        nh, nw = round(h*ratio), round(w*ratio)
        top, left = round((th-nh)/2-.1), round((tw-nw)/2-.1)
        # Training data may already be augmented; don't stretch the tensor.
        resized = F.interpolate(image[None].float(), size=(nh,nw), mode="bilinear", align_corners=False)[0].round().to(image.dtype)
        sample["img"] = F.pad(resized, (left,tw-nw-left,top,th-nh-top), value=114)
        boxes = sample["bboxes"].clone()
        if len(boxes):
            boxes[:,0] = (boxes[:,0]*w*ratio+left)/tw
            boxes[:,1] = (boxes[:,1]*h*ratio+top)/th
            boxes[:,2] *= w*ratio/tw
            boxes[:,3] *= h*ratio/th
        sample["bboxes"] = boxes
        if sample.get("ratio_pad"):
            gain, pad = sample["ratio_pad"]
            sample["ratio_pad"] = ((gain[0]*ratio,gain[1]*ratio), (pad[0]*ratio+left,pad[1]*ratio+top))
        sample["resized_shape"] = self.hw
        return sample


def reset_replaced_rows(model, indices):
    """Only final classifier rows change; retained weights/biases stay bit-identical."""
    import torch
    import torch.nn as nn
    head = model.model[-1]
    if head.nc != 80 or len(head.cv3) != 3:
        raise ValueError("replace_coco requires a compatible 80-class YOLOv8 head")
    import math
    for branch_index,branch in enumerate(head.cv3):
        conv = branch[-1]
        if not isinstance(conv, nn.Conv2d) or conv.out_channels != 80:
            raise ValueError("Unexpected classifier topology")
        replacement = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, bias=conv.bias is not None).to(conv.weight.device)
        with torch.no_grad():
            conv.weight[indices] = replacement.weight[indices]
            if conv.bias is not None:
                conv.bias[indices] = math.log(5 / 80 / (640/float(head.stride[branch_index]))**2)


def train(cfg, run):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    verify_manifest(run)
    options = cfg["train"]
    device = choose_device(options["device"])
    batch = options["batch"]
    if batch == "auto":
        batch = -1 if device != "cpu" else 4
    workers = min(4, max(0, (os.cpu_count() or 2)//2)) if options["workers"] == "auto" else options["workers"]
    hw = cfg.hw("train")
    replaced = cfg.classes["replaced"]
    source = cfg.path(cfg["model"]["pretrained"])
    resume = cfg.path(options["resume_from"])
    if source is not None and not source.is_file() and resume is None:
        raise ValueError(f"Pretrained weights missing: {source}; run download_assets.sh or use pretrained: null")
    yolo = YOLO(str(resume or source) if (resume or source) else "yolov8n.yaml")
    from ultralytics.nn.modules.head import Detect
    if len(yolo.model.model)!=23 or not isinstance(yolo.model.model[-1],Detect) or yolo.model.model[0].conv.out_channels!=16:
        raise ValueError("Configured yolov8n_standard requires the standard YOLOv8n topology")
    if replaced and not resume:
        from .classes import COCO
        source_names = list(yolo.model.names.values()) if isinstance(yolo.model.names, dict) else yolo.model.names
        if source is None or source_names != COCO:
            raise ValueError("replace_coco initialization must have the standard 80 COCO class names")

    class ProjectTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if replaced and not resume:
                if weights is None:
                    raise ValueError("replace_coco requires pretrained initialization")
                reset_replaced_rows(model, replaced)
            return model

        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode=="val":
                # Remove the upstream variable rectangular-batch padding before
                # applying our fixed deployment aspect ratio.
                dataset.rect=False
                dataset.transforms=dataset.build_transforms(deepcopy(self.args))
            dataset.transforms.append(FixedInput(hw))
            # close_mosaic rebuilds transforms; preserve the fixed-input transform.
            original = dataset.close_mosaic
            def close_mosaic(hyp):
                original(hyp)
                dataset.transforms.append(FixedInput(hw))
            dataset.close_mosaic = close_mosaic
            return dataset

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if tuple(batch["img"].shape[-2:]) != hw:
                raise RuntimeError(f"Trainer ignored input_hw; got {batch['img'].shape}")
            return batch

    allowed = {"epochs", "patience", "optimizer", "lr0", "weight_decay", "deterministic", "freeze",
               "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "mosaic", "close_mosaic", "mixup"}
    arguments = {k: options[k] for k in allowed}
    arguments.update(data=str(run/"prepare/dataset.yaml"), imgsz=max(hw), batch=batch, device=device,
                     workers=workers, amp=(device != "cpu") if options["amp"] == "auto" else options["amp"],
                     seed=cfg["project"]["seed"], project=str(run), name="train", exist_ok=False,
                     plots=True, save=True, val=True, multi_scale=False)
    target = run/"train"
    if resume:
        if not resume.is_file():
            raise ValueError(f"Resume checkpoint missing: {resume}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
            raise ValueError("Resume requires an unfinished last.pt with optimizer state")
        if Path(checkpoint["train_args"]["data"]).resolve() != (run/"prepare/dataset.yaml").resolve():
            raise ValueError("Resume in the original --run-id so dataset and optimizer progress stay linked")
        # Upstream restores original args; reject mismatched data/classes/shape explicitly.
        resumed_model = checkpoint.get("ema") or checkpoint.get("model")
        resumed_names = list(resumed_model.names.values()) if isinstance(resumed_model.names, dict) else resumed_model.names
        if resumed_names != cfg.classes["names"]:
            raise ValueError("Resume class names differ from this run")
        arguments.update(resume=str(resume))
    yolo.train(trainer=ProjectTrainer, **arguments)
    actual = Path(yolo.trainer.save_dir).resolve()
    best = actual/"weights/best.pt"
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt")
    write_json(run/"training.json", {"best": str(best), "sha256": digest(best), "save_dir": str(actual),
                "input_hw": hw, "device": device, "batch_requested": batch,
                "batch_actual": yolo.trainer.batch_size, "environment": environment(),
                "retained_class_preservation": "requires independent regression; no guarantee from initialization"})
    return run/"training.json"
