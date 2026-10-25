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


def transfer_k1_fast_pretrained(model, checkpoint):
    """Transfer matching stem/head tensors, never unrelated backbone layers."""
    from ultralytics import YOLO
    source = YOLO(str(checkpoint)).model.cpu().float()
    if source.yaml.get("architecture") == "yolov8n_k1_fast":
        model.load(source)
        return "K1-fast checkpoint: compatible tensors"
    source_stem = source.model[0].conv
    target_stem = model.model[0].conv
    if source_stem.state_dict().keys() != target_stem.state_dict().keys():
        raise ValueError("Standard checkpoint stem layout differs from the K1-fast stem")
    if any(source_stem.state_dict()[k].shape != target_stem.state_dict()[k].shape for k in source_stem.state_dict()):
        raise ValueError("Standard checkpoint stem channels differ from the K1-fast architecture")
    target_stem.load_state_dict(source_stem.state_dict())
    copied = ["model.0.conv"]
    source_head, target_head = source.model[-1], model.model[-1]
    if source_head.nc == 80 and target_head.nc in {1, 80} and source_head.nl == target_head.nl == 3:
        for branch_name in ["cv2", "cv3"]:
            for branch_index, (source_branch, target_branch) in enumerate(zip(
                    getattr(source_head, branch_name), getattr(target_head, branch_name))):
                target_modules = dict(target_branch.named_modules())
                for name, source_module in source_branch.named_modules():
                    if not name or any(source_module.children()):
                        continue
                    target_module = target_modules.get(name)
                    if target_module is None or type(target_module) is not type(source_module):
                        continue
                    source_state, target_state = source_module.state_dict(), target_module.state_dict()
                    if source_state.keys() == target_state.keys() and all(
                            source_state[key].shape == target_state[key].shape for key in source_state):
                        target_module.load_state_dict(source_state)
                        copied.append(f"head.{branch_name}.{branch_index}.{name}")
    return f"standard YOLO checkpoint: transferred {len(copied)} matching stem/head modules"


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
    architecture = cfg["model"]["architecture"]
    if architecture == "yolov8n_k1_fast":
        from .architectures import fast_model_yaml, register_k1_fast_modules
        register_k1_fast_modules()
        yolo = YOLO(str(resume) if resume else fast_model_yaml())
    else:
        yolo = YOLO(str(resume or source) if (resume or source) else "yolov8n.yaml")
    from ultralytics.nn.modules.head import Detect
    if not isinstance(yolo.model.model[-1],Detect) or yolo.model.model[0].conv.out_channels!=16:
        raise ValueError(f"Configured {architecture} has an unexpected detector topology")
    if architecture == "yolov8n_standard" and len(yolo.model.model) != 23:
        raise ValueError("Configured yolov8n_standard requires the standard YOLOv8n topology")
    if architecture == "yolov8n_k1_fast" and len(yolo.model.model) != 31:
        raise ValueError("Configured yolov8n_k1_fast requires the 31-layer reference approximation")
    if replaced and not resume:
        from .classes import COCO
        if source is None:
            raise ValueError("replace_coco initialization requires the standard 80-class checkpoint")
        source_model = yolo.model if architecture == "yolov8n_standard" else YOLO(str(source)).model
        source_names = list(source_model.names.values()) if isinstance(source_model.names, dict) else source_model.names
        if source_names != COCO:
            raise ValueError("replace_coco initialization must have the standard 80 COCO class names")

    pretrained_transfer = ("resume checkpoint" if resume else
                           "standard Ultralytics pretrained checkpoint" if architecture == "yolov8n_standard" and source else
                           "random initialization")

    class ProjectTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            nonlocal pretrained_transfer
            model = super().get_model(cfg, weights, verbose)
            if architecture == "yolov8n_k1_fast" and source is not None and not resume:
                # Match by semantic detector branch and exact tensor shape;
                # other backbone layers use different topology and stay fresh.
                pretrained_transfer = transfer_k1_fast_pretrained(model, source)
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
                "scheme": cfg["workflow"]["scheme"], "architecture": architecture,
                "input_hw": hw, "device": device, "batch_requested": batch,
                "batch_actual": yolo.trainer.batch_size, "environment": environment(),
                "pretrained_transfer": pretrained_transfer,
                "retained_class_preservation": "requires independent regression; no guarantee from initialization"})
    return run/"training.json"
