#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$1"; shift
KIND=train
[[ "$STAGE" != quantize ]] || KIND=quant
PYTHON_BIN="${TRAIN_PYTHON:-}"
[[ "$KIND" != quant ]] || PYTHON_BIN="${QUANT_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ ! -f "$PROJECT_ROOT/.environments.json" ]]; then
    echo 'Run bash scripts/setup_env.sh first.' >&2; exit 1
  fi
  PYTHON_BIN="$(python3 - "$PROJECT_ROOT/.environments.json" "$KIND" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]+'_python'])
PY
)"
fi
cd -- "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_ROOT/.cache/ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
export YOLO_AUTOINSTALL=false
export MPLBACKEND=Agg
exec "$PYTHON_BIN" -m k1_yolo.cli "$STAGE" --root "$PROJECT_ROOT" "$@"
