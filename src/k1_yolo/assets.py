from __future__ import annotations
import os
from pathlib import Path
import time
import urllib.request
from .common import digest,read_json,write_json


def download(cfg):
    target=cfg.path(cfg["model"]["pretrained"])
    if target is None:
        print("Random initialization selected; no pretrained download")
        return
    assets=read_json(cfg.root/"configs/assets.json")["assets"]
    asset=assets.get(target.name)
    if target.is_file():
        if asset and digest(target)!=asset["sha256"]:
            raise ValueError(f"Existing {target.name} differs from pinned official weight; use a different filename for custom checkpoints")
        print("Weights present:",target,"SHA256",digest(target))
        return
    if not asset:
        raise ValueError(f"No pinned asset for {target.name}; supply the local checkpoint")
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(asset["url"],timeout=60) as response, temporary.open("wb") as out:
                while block:=response.read(1024*1024):
                    out.write(block)
            if temporary.stat().st_size!=asset["bytes"] or digest(temporary)!=asset["sha256"]:
                raise ValueError("Downloaded weight failed size/SHA256 verification")
            os.replace(temporary,target)
            write_json(target.with_suffix(".source.json"),asset)
            print("Downloaded verified weights:",target)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt==2:
                raise
            time.sleep(attempt+1)
