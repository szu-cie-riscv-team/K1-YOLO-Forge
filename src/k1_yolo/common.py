from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as out:
        json.dump(value, out, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        out.write("\n")
        temporary = out.name
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    versions = {}
    for name in ["k1-yolo", "torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "xslim", "numpy", "opencv-python", "PyYAML"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": versions}


def run_logged(command, logfile, *, env=None, cwd=None):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    with logfile.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"argv": list(map(str, command))}) + "\n")
        stream.flush()
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env, cwd=cwd)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if status:
        raise RuntimeError(f"Command exited {status}; see {logfile}")
