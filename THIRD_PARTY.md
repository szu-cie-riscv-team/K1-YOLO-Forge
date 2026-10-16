# Dependencies and model provenance

- This repository is distributed under AGPL-3.0; see LICENSE.
- Ultralytics 8.3.200: https://github.com/ultralytics/ultralytics/tree/v8.3.200 — AGPL-3.0 / vendor licensing options. Used as the training implementation; its original source is installed as a dependency.
- Official YOLOv8n pretrained weight: pinned URL, byte count and SHA-256 in configs/assets.json. Licensing: https://www.ultralytics.com/license . Downloaded weights are not committed.
- SpacemiT XSlim 2.1.2: https://github.com/spacemit-com/xslim — Apache-2.0. Invoked as the official quantization tool; no private replacement quantizer is included.
- PyTorch/torchvision: https://pytorch.org/get-started/previous-versions/ . CPU/cu126 wheels are downloaded from the official PyTorch index.
- ONNX and ONNX Runtime: https://onnx.ai/ and https://onnxruntime.ai/ . Host CPU runtime is distinct from the official SpacemiT board runtime.
- Historical reference models were inspected read-only from the user's archive. They are not included or downloaded by this repository. Inspection evidence is in docs/research/reference-models.json.
- Users provide dataset content and should retain its source/license information with the dataset manifest.

Resolved dependency versions are written into each environment and run. See docs/implementation-validation.md for actually exercised combinations.
