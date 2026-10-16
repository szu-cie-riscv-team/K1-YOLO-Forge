from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml

from .common import digest, fingerprint, read_json, write_json
from .preprocess import letterbox, read_image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def read_labels(path, mapping):
    if not path.is_file():
        raise ValueError(f"Missing label (confirmed background needs an empty file): {path}")
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected detection TXT with 5 columns: {path}:{index}")
        values = np.asarray([float(x) for x in parts], dtype=float)
        cls, x, y, w, h = values
        if not np.isfinite(values).all() or cls != int(cls) or not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1:
            raise ValueError(f"Invalid normalized box/class: {path}:{index}")
        if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
            raise ValueError(f"Box extends beyond image: {path}:{index}")
        cls = int(cls)
        if cls not in mapping:
            raise ValueError(f"Unknown annotation class {cls}: {path}:{index}")
        if mapping[cls] is not None:
            result.append([mapping[cls], x, y, w, h])
    return result


def scan_dataset(root, mapping, source, provided, split_by):
    root = Path(root)
    if not (root / "images").is_dir():
        raise ValueError(f"Dataset requires images/ and labels/: {root}")
    groups = {}
    if (root / "groups.csv").exists():
        with (root / "groups.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"image", "group"} <= set(reader.fieldnames or []):
                raise ValueError(f"{root}/groups.csv needs image,group columns")
            for item in reader:
                key = item["image"].replace("\\", "/").removeprefix("./")
                if key in groups or not item["group"].strip():
                    raise ValueError("Duplicate image or empty group in groups.csv")
                groups[key] = item["group"].strip()
    rows = []
    for path in images(root / "images"):
        relative = path.relative_to(root / "images")
        subset = relative.parts[0] if provided else None
        if provided and subset not in {"train", "val", "test"}:
            raise ValueError(f"Provided split requires images/train, images/val, optional images/test: {path}")
        label_path = (root / "labels" / relative).with_suffix(".txt")
        labels = read_labels(label_path, mapping)
        image = read_image(path)
        key = path.relative_to(root).as_posix()
        group = groups.get(key)
        if split_by == "capture_group" and not group:
            raise ValueError(f"Missing groups.csv entry for {key}; supply capture groups or explicitly set split_by=image")
        pixels = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
        dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")
        rows.append({"path": str(path.resolve()), "label_path": str(label_path.resolve()), "source": source,
                     "group": group or pixels, "split": subset, "labels": labels,
                     "sha256": digest(path), "label_sha256": digest(label_path), "pixel_hash": pixels,
                     "dhash": dhash, "shape_hw": list(image.shape[:2])})
    if not rows:
        raise ValueError(f"No images found: {root}")
    return rows


def split_pool(rows, counts, rng):
    # Allocate whole groups to a single subset. Exact counts use a subset of a
    # group's images when needed; discarded images are never assigned elsewhere.
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    bundles = list(groups.values())
    rng.shuffle(bundles)
    result = []
    # Smaller sets first leaves the largest remaining pool for train.
    for subset in sorted(counts, key=lambda s: (counts[s], s)):
        remaining = counts[subset]
        while remaining:
            if not bundles:
                raise ValueError(f"Cannot fill {subset} with disjoint capture groups; {remaining} images short")
            bundle = bundles.pop()
            rng.shuffle(bundle)
            selected = bundle[:remaining]
            for row in selected:
                result.append(dict(row, split=subset))
            remaining -= len(selected)
    return result


def check_disjoint(rows):
    by_pixels, by_group = {}, {}
    for row in rows:
        for key, table in [(row["pixel_hash"], by_pixels), (row["group"], by_group)]:
            if key in table and table[key]["split"] != row["split"]:
                raise ValueError(f"Cross-split duplicate/capture group: {row['path']} and {table[key]['path']}")
            table[key] = row
    near = []
    # Bounded diagnostic: near-duplicates are review candidates, not automatic labels.
    for i, a in enumerate(rows):
        if len(near) >= 30:
            break
        for b in rows[i+1:]:
            if a["split"] != b["split"] and (a["dhash"] ^ b["dhash"]).bit_count() <= 2:
                near.append([a["path"], b["path"]])
                if len(near) >= 30:
                    break
    return near


def prepare(cfg, run):
    destination = run / "prepare"
    destination.mkdir(parents=True, exist_ok=True)
    mapping = cfg.classes
    provided = cfg["data"]["split"]["mode"] == "provided"
    root = cfg.path(cfg["data"]["source_dir"])
    names_path = root / "classes.txt"
    if names_path.exists():
        names = [s.strip() for s in names_path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        expected = [mapping["names"][mapping["custom_annotation_to_output"][i]] for i in range(len(mapping["custom_annotation_to_output"]))]
        if names != expected:
            raise ValueError(f"classes.txt order differs from config: {names} != {expected}")
    custom = scan_dataset(root, mapping["custom_annotation_to_output"], "custom", provided, cfg["data"]["split_by"])
    all_rows = custom.copy()
    rng = random.Random(cfg["project"]["seed"])
    if mapping["mode"] == "replace_coco":
        options = cfg["classes"]["replace_coco"]
        retained_map = {i: i if i in mapping["retained"] else None for i in range(80)}
        # Replay and retained_val are separate flat YOLO-format directories.
        replay = scan_dataset(cfg.path(options["replay_data"]), retained_map, "replay", False, cfg["data"]["split_by"])
        retained_val = scan_dataset(cfg.path(options["retained_validation_data"]), retained_map, "retained_val", False, cfg["data"]["split_by"])
        replay = [r for r in replay if r["labels"]]
        retained_val = [dict(r, split="val") for r in retained_val if r["labels"]]
        if not replay or not retained_val:
            raise ValueError("replace_coco needs nonempty retained-class replay and validation data")
        if not provided:
            raise ValueError("replace_coco currently requires provided custom train/val splits")
        ratio = options["replay_train_fraction"]
        needed = round(sum(r["split"] == "train" for r in custom) * ratio / (1-ratio))
        if needed > len(replay):
            raise ValueError(f"Replay requires {needed} unique images, only {len(replay)} available")
        replay = [dict(r, split="train") for r in rng.sample(replay, needed)]
        all_rows += replay + retained_val
    if provided:
        selected = all_rows
        actual = Counter(r["split"] for r in selected)
        for subset, expected in cfg["data"]["split"]["expected_counts"].items():
            if expected is not None and actual[subset] != expected:
                raise ValueError(f"Expected final mixed {subset}={expected}, found {actual[subset]}")
    else:
        mode = cfg["data"]["split"]["mode"]
        if mode == "counts":
            counts = cfg["data"]["split"]["counts"]
        else:
            ratios = cfg["data"]["split"]["ratios"]
            counts = {s: math.floor(len(all_rows) * ratios[s]) for s in ["val", "test"]}
            counts["train"] = len(all_rows) - sum(counts.values())
        selected = split_pool(all_rows, counts, rng)
    if not all(any(r["split"] == split for r in selected) for split in ["train", "val"]):
        raise ValueError("Both train and val must contain images")
    near = check_disjoint(selected)
    coverage, source_counts = {}, {}
    for subset in ["train", "val", "test"]:
        items = [r for r in selected if r["split"] == subset]
        boxes = Counter(int(box[0]) for r in items for box in r["labels"])
        coverage[subset] = {name: boxes[i] for i, name in enumerate(mapping["names"])}
        source_counts[subset] = dict(Counter(r["source"] for r in items))
        if subset in {"train", "val"}:
            missing = [name for i, name in enumerate(mapping["names"]) if not boxes[i]]
            if missing:
                raise ValueError(f"No {subset} labels for classes: {missing}")
    calibration = []
    if cfg["data"]["calibration_source"] == "train":
        calibration = [dict(path=r["path"], sha256=r["sha256"], pixel_hash=r["pixel_hash"]) for r in selected if r["split"] == "train"]
    else:
        forbidden = {r["pixel_hash"] for r in selected if r["split"] in {"val", "test"}}
        seen = set()
        for path in images(cfg.path(cfg["data"]["calibration_dir"])):
            image = read_image(path)
            pixels = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            if pixels in forbidden:
                raise ValueError(f"Calibration overlaps val/test: {path}")
            if pixels in seen:
                raise ValueError(f"Duplicate calibration image pixels: {path}")
            seen.add(pixels)
            calibration.append(dict(path=str(path.resolve()), sha256=digest(path), pixel_hash=pixels))
    if not calibration:
        raise ValueError("No calibration images")
    rng.shuffle(calibration)
    requested = cfg["quantize"]["calibration"]["max_samples"]
    calibration = calibration[:requested]
    warnings = []
    if len(calibration) < requested:
        warnings.append(f"Requested up to {requested} calibration images, using {len(calibration)} distinct images")
    if near:
        warnings.append("Review near_duplicate_candidates; perceptual hashes are not proof of leakage")
    if cfg["data"]["split_by"] == "image":
        warnings.append("Capture groups were not checked (split_by=image explicitly selected)")
    if not any(r["split"] == "test" for r in selected):
        warnings.append("No independent test set; validation is not final independent acceptance")
    dataset_root = destination / "dataset"
    for index, row in enumerate(selected):
        uid = fingerprint([row["source"], row["path"]])[:16]
        image_target = dataset_root / "images" / row["split"] / (uid+Path(row["path"]).suffix.lower())
        label_target = dataset_root / "labels" / row["split"] / (uid+".txt")
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], image_target)
        label_target.write_text("".join(f"{int(box[0])} " + " ".join(f"{v:.9g}" for v in box[1:])+"\n" for box in row["labels"]), encoding="utf-8")
        row["prepared_image"] = str(image_target.resolve())
        row["prepared_label"] = str(label_target.resolve())
        row["prepared_label_sha256"] = digest(label_target)
        if index < 8:
            preview = cv2.cvtColor((letterbox(read_image(row["path"]), cfg.hw("train"))[0][0].transpose(1,2,0)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            _, meta = letterbox(read_image(row["path"]), cfg.hw("train"))
            sh, sw = row["shape_hw"]
            for cls, x, y, w, h in row["labels"]:
                x1 = round((x-w/2)*sw*meta["ratio"]+meta["left"])
                y1 = round((y-h/2)*sh*meta["ratio"]+meta["top"])
                x2 = round((x+w/2)*sw*meta["ratio"]+meta["left"])
                y2 = round((y+h/2)*sh*meta["ratio"]+meta["top"])
                cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(preview, mapping["names"][int(cls)], (x1,max(10,y1)), cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
            folder = destination/"previews";folder.mkdir(exist_ok=True)
            cv2.imwrite(str(folder/f"{index:02d}.jpg"), preview)
    dataset_yaml = {"path": str(dataset_root.resolve()), "train": "images/train", "val": "images/val", "names": mapping["names"], "nc": mapping["nc"]}
    if any(r["split"] == "test" for r in selected):
        dataset_yaml["test"] = "images/test"
    (destination/"dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True), encoding="utf-8")
    report = {"images": dict(Counter(r["split"] for r in selected)), "sources": source_counts,
              "boxes_per_class": coverage, "calibration_images": len(calibration), "warnings": warnings,
              "near_duplicate_candidates": near, "unused_images": sorted(set(r["path"] for r in all_rows)-set(r["path"] for r in selected))}
    write_json(destination/"class_map.json", mapping)
    write_json(destination/"manifest.json", {"images": selected, "calibration": calibration})
    write_json(destination/"report.json", report)
    print("Data prepared:", report["images"], "; calibration:", len(calibration), flush=True)
    for warning in warnings:
        print("NOTICE:", warning, flush=True)
    return destination/"manifest.json"


def verify_manifest(run):
    manifest = read_json(run/"prepare/manifest.json")
    for row in manifest["images"]:
        for path, expected in [(row["path"],row["sha256"]), (row["label_path"],row["label_sha256"]),
                               (row["prepared_image"],row["sha256"]), (row["prepared_label"],row["prepared_label_sha256"])]:
            if not Path(path).is_file() or digest(path) != expected:
                raise ValueError(f"Dataset changed since prepare: {path}; prepare a new run")
    for row in manifest["calibration"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError(f"Calibration changed since prepare: {row['path']}")
    return manifest
