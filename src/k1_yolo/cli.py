from __future__ import annotations

import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import uuid

from .common import digest,environment,fingerprint,read_json,write_json,run_logged
from .config import Config

STAGES=["prepare","train","export","quantize","evaluate","package"]


def source_key(cfg):
    roots=[cfg.path(cfg["data"]["source_dir"])]
    if cfg["data"]["calibration_source"]=="directory":
        roots.append(cfg.path(cfg["data"]["calibration_dir"]))
    if cfg.classes["mode"]=="replace_coco":
        roots.extend(cfg.path(cfg["classes"]["replace_coco"][k]) for k in ["replay_data","retained_validation_data"])
    records=[]
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Dataset directory missing: {root}")
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".webp",".txt",".csv"}:
                records.append((str(p),digest(p)))
    return fingerprint(records)


def stage_key(cfg,stage,state,args,memo=None):
    memo={} if memo is None else memo
    if stage in memo:return memo[stage]
    def remember(value):
        memo[stage]=value
        return value
    if stage=="prepare":
        code=fingerprint([(p.name,digest(p)) for p in sorted((cfg.root/"src/k1_yolo").glob("*.py"))])
        return remember(fingerprint([code,cfg["data"],cfg["classes"],cfg["project"]["seed"],cfg.hw("train"),
                            cfg["quantize"]["calibration"]["max_samples"],source_key(cfg)]))
    dependencies={"train":["prepare"],"export":["prepare","train"],"quantize":["prepare","export"],
                  "evaluate":["prepare","export","quantize"],"package":["prepare","export","quantize","evaluate"]}[stage]
    checkpoint_arg=args.checkpoint or state.get("export",{}).get("external_checkpoint")
    if stage=="export" and checkpoint_arg:
        dependencies=["prepare"]
    dep=[]
    for name in dependencies:
        item=state.get(name)
        if not item or item.get("status")!="PASS":
            raise ValueError(f"{stage} requires successful {name}; run that stage first")
        if stage_key(cfg,name,state,args,memo)!=item["key"]:
            raise ValueError(f"{name} inputs/config changed; rerun {name} in a new run or with --force")
        if not Path(item["artifact"]).exists() or digest(item["artifact"])!=item["sha256"]:
            raise ValueError(f"{name} artifact changed since validation")
        dep.append((name,item["key"],item["sha256"]))
    if stage=="train":
        path=cfg.path(cfg["train"]["resume_from"] or cfg["model"]["pretrained"])
        consumed=state.get("train",{}).get("resume_input")
        sha=(consumed["sha256"] if consumed and consumed["path"]==str(path) and state["train"].get("status")=="PASS"
             else digest(path) if path and path.is_file() else None)
        return remember(fingerprint([dep,cfg["train"],cfg["model"],cfg.hw("train"),str(path),sha]))
    if stage=="export":
        checkpoint=Path(checkpoint_arg).resolve() if checkpoint_arg else Path(read_json(Path(state["train"]["artifact"]))["best"])
        return remember(fingerprint([dep,cfg["export"],cfg.hw("export"),digest(checkpoint)]))
    if stage=="quantize":
        return remember(fingerprint([dep,cfg["quantize"],cfg.hw("export")]))
    if stage=="evaluate":
        return remember(fingerprint([dep,cfg["evaluation"],cfg["deploy"],args.split]))
    return remember(fingerprint([dep,cfg["deploy"],cfg["preprocess"]]))


def context(cfg,args):
    base=cfg.path(cfg["project"]["output_dir"])/cfg["project"]["name"]
    if args.run_id:
        run_id=args.run_id
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}",run_id):
            raise ValueError("Invalid --run-id")
    elif args.command in {"prepare","pipeline"} and (args.command!="pipeline" or args.from_stage=="prepare"):
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    else:
        if not (base/"latest.json").exists():
            raise ValueError("No prepared run; run prepare_data.sh first")
        run_id=read_json(base/"latest.json")["run_id"]
    run=base/run_id;run.mkdir(parents=True,exist_ok=True)
    if not (run/"config.initial.json").exists():
        write_json(run/"config.initial.json",cfg.raw)
    write_json(run/"config.resolved.json",cfg.raw)
    write_json(base/"latest.json",{"run_id":run_id})
    return run


def valid_artifact(item,stage):
    path=Path(item.get("artifact",""))
    if not path.is_file() or digest(path)!=item.get("sha256"):
        return False
    if stage=="train":
        record=read_json(path);best=Path(record["best"])
        return best.is_file() and digest(best)==record["sha256"]
    if stage=="package":
        record=read_json(path);archive=Path(record["archive"]);folder=Path(record["directory"])
        if not archive.is_file() or digest(archive)!=record["sha256"] or not (folder/"manifest.json").is_file():
            return False
        manifest=read_json(folder/"manifest.json")
        return all((folder/name).is_file() and digest(folder/name)==sha for name,sha in manifest["files"].items())
    return True


def execute(cfg,run,stage,args):
    import fcntl
    lock=(run/".lock").open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another process is using run {run.name}")
    state=read_json(run/"state.json") if (run/"state.json").exists() else {}
    state_name="evaluate_test" if stage=="evaluate" and args.split=="test" else stage
    key=stage_key(cfg,stage,state,args)
    previous=state.get(state_name)
    external_checkpoint=args.checkpoint or state.get("export",{}).get("external_checkpoint")
    if previous and previous.get("status")=="PASS" and previous["key"]==key and valid_artifact(previous,stage):
        from .data import verify_manifest
        if stage!="prepare":
            verify_manifest(run)
        print(f"CACHED {stage}: {previous['artifact']}",flush=True)
        return
    if previous and previous.get("status")=="PASS" and not args.force:
        raise ValueError(f"{stage} already succeeded with different inputs; use a new --run-id or --force (archives prior stage)")
    existing=run/{"evaluate":"evaluation"}.get(stage,stage)
    resuming=stage=="train" and cfg["train"]["resume_from"] and cfg.path(cfg["train"]["resume_from"]).is_relative_to(existing)
    if existing.exists() and not resuming and state_name!="evaluate_test":
        history=run/"history"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:6])
        history.mkdir(parents=True);shutil.move(existing,history/stage)
    state[state_name]={"status":"RUNNING","key":key,"environment":environment()}
    if stage=="train" and cfg["train"]["resume_from"]:
        path=cfg.path(cfg["train"]["resume_from"])
        state[state_name]["resume_input"]={"path":str(path),"sha256":digest(path)}
    if stage=="export" and external_checkpoint:
        state[state_name]["external_checkpoint"]=str(Path(external_checkpoint).resolve())
    write_json(run/"state.json",state)
    try:
        if stage=="prepare":
            from .data import prepare
            result=prepare(cfg,run)
        elif stage=="train":
            from .training import train
            result=train(cfg,run)
        elif stage=="export":
            from .exporting import export
            result=export(cfg,run,external_checkpoint)
        elif stage=="quantize":
            from .quantization import quantize
            result=quantize(cfg,run)
        elif stage=="evaluate":
            from .evaluation import evaluate
            result=evaluate(cfg,run,args.split)
        else:
            from .packaging import package
            result=package(cfg,run)
        state[state_name].update(status="PASS",artifact=str(result),sha256=digest(result))
        write_json(run/"state.json",state)
        print(f"PASS {stage}: {result}",flush=True)
    except BaseException as exc:
        state[state_name].update(status="FAIL",error=str(exc))
        write_json(run/"state.json",state)
        raise
    finally:
        lock.close()


def doctor(cfg):
    result=environment()
    import torch
    result["cuda_available"]=torch.cuda.is_available()
    if torch.cuda.is_available():
        a=torch.ones((16,16),device="cuda");(a@a).sum().item()
        result["gpu"]=torch.cuda.get_device_name(0)
    result["train_input_hw"]=cfg.hw("train");result["export_input_hw"]=cfg.hw("export")
    result["classes"]=cfg.classes
    result["free_disk_gib"]=shutil.disk_usage(cfg.root).free/(1024**3)
    write_json(cfg.path(cfg["project"]["output_dir"])/"environment.json",result)
    print(result)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Linux/WSL YOLO → official XSlim → K1 model package")
    parser.add_argument("command",choices=STAGES+["pipeline","download","doctor"])
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--root",default=str(Path.cwd()))
    parser.add_argument("--run-id")
    parser.add_argument("--force",action="store_true",help="archive and rerun a completed stage")
    parser.add_argument("--from",dest="from_stage",choices=STAGES,default="prepare")
    parser.add_argument("--to",dest="to_stage",choices=STAGES,default="package")
    parser.add_argument("--checkpoint",help="export a compatible checkpoint without running train")
    parser.add_argument("--split",choices=["val","test"],default="val")
    args=parser.parse_args(argv)
    try:
        cfg=Config(args.config,args.root)
        if args.command=="download":
            from .assets import download
            download(cfg);return
        if args.command=="doctor":
            doctor(cfg);return
        run=context(cfg,args)
        print("RUN",run.name,flush=True)
        if args.command!="pipeline":
            execute(cfg,run,args.command,args);return
        if args.split!="val":
            raise ValueError("Pipeline uses validation; run evaluate.sh --split test separately for independent testing")
        start,end=STAGES.index(args.from_stage),STAGES.index(args.to_stage)
        if start>end:
            raise ValueError("--from must not follow --to")
        registry=read_json(cfg.root/".environments.json") if (cfg.root/".environments.json").exists() else {}
        for stage in STAGES[start:end+1]:
            python=registry.get("quant_python") if stage=="quantize" else sys.executable
            if not python:
                raise ValueError("Run setup_env.sh --profile cpu/cuda/auto to configure the XSlim environment")
            command=[python,"-m","k1_yolo.cli",stage,"--config",str(cfg.filename),"--root",str(cfg.root),"--run-id",run.name,"--split",args.split]
            if args.force:command.append("--force")
            if args.checkpoint:command.extend(["--checkpoint",args.checkpoint])
            env=os.environ.copy();env.setdefault("OMP_NUM_THREADS","4");env.setdefault("MKL_NUM_THREADS","4")
            run_logged(command,run/"logs"/(stage+".log"),env=env,cwd=cfg.root)
        print("Pipeline completed:",run,flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        if os.environ.get("K1_YOLO_TRACEBACK")=="1":
            traceback.print_exc()
        raise SystemExit(1)


if __name__=="__main__":
    main()
