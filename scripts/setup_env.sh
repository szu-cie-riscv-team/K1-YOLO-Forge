#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_ROOT/.cache/ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
export YOLO_AUTOINSTALL=false
export MPLBACKEND=Agg
PROFILE=auto
INSTALL_SYSTEM=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_ROOT="${ENV_ROOT:-$PROJECT_ROOT/.venvs}"
while (($#)); do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --env-root) ENV_ROOT="$2"; shift 2 ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    --help) echo 'setup_env.sh [--profile auto|cpu|cuda|inspect] [--env-root PATH] [--install-system]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux ]] || { echo 'Linux/WSL is required.' >&2; exit 1; }
if ((INSTALL_SYSTEM)); then
  command -v apt-get >/dev/null || { echo '--install-system supports Debian/Ubuntu apt.' >&2; exit 1; }
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git ca-certificates libgl1 libglib2.0-0
fi
mkdir -p "$ENV_ROOT"
ENV_ROOT="$(cd -- "$ENV_ROOT" && pwd)"
"$PYTHON_BIN" -c 'import sys; assert (3,10)<=sys.version_info<(3,13), "Use Python 3.10-3.12 to bootstrap"'
if [[ "$PROFILE" != inspect ]] && ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info>=(3,11)' 2>/dev/null; then
  # XSlim's ORT 1.24 wheels require Python >=3.11 on Linux.
  if ! command -v uv >/dev/null 2>&1; then
    "$PYTHON_BIN" -m venv "$ENV_ROOT/bootstrap" || { echo 'Install python3-venv/python3-pip first.' >&2; exit 1; }
    "$ENV_ROOT/bootstrap/bin/python" -m pip install uv==0.12.21
    export PATH="$ENV_ROOT/bootstrap/bin:$PATH"
  fi
  uv python install 3.11
  PYTHON_BIN="$(uv python find 3.11)"
fi
if [[ "$PROFILE" == auto ]]; then
  PROFILE=cpu
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then PROFILE=cuda; fi
fi
case "$PROFILE" in cpu|cuda|inspect) ;; *) echo 'Invalid profile.' >&2; exit 2 ;; esac
if [[ "$PROFILE" != inspect && "$(uname -m)" != x86_64 ]]; then
  echo 'Training/XSlim bootstrap supports x86_64 hosts. K1 is the deployment target.' >&2; exit 1
fi
create_env() {
  local target="$1"
  if [[ ! -x "$target/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then uv venv --python "$PYTHON_BIN" "$target"
    elif ! "$PYTHON_BIN" -m venv "$target"; then
      echo 'Install python3-venv/python3-pip (or rerun with --install-system).' >&2; exit 1
    fi
  fi
}
install() {
  local target="$1"; shift
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$target/bin/python" "$@"
  else "$target/bin/python" -m pip install "$@"; fi
}
if [[ "$PROFILE" == inspect ]]; then
  INSPECT_ENV="${VENV_PATH:-$ENV_ROOT/inspect}"
  create_env "$INSPECT_ENV"
  install "$INSPECT_ENV" -r "$PROJECT_ROOT/requirements/inspect.txt"
  "$PYTHON_BIN" - "$PROJECT_ROOT/.environments.json" "$INSPECT_ENV/bin/python" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);data=json.loads(p.read_text()) if p.exists() else {}
data['inspect_python']=str(Path(sys.argv[2]).absolute());p.write_text(json.dumps(data,indent=2)+'\n')
PY
  echo "Inspection environment: $INSPECT_ENV"; exit 0
fi
TORCH_INDEX=https://download.pytorch.org/whl/cpu
[[ "$PROFILE" != cuda ]] || TORCH_INDEX=https://download.pytorch.org/whl/cu126
for KIND in train quant; do
  TARGET="$ENV_ROOT/$KIND-$PROFILE"
  create_env "$TARGET"
  install "$TARGET" torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX"
  LOCK="$PROJECT_ROOT/requirements/$KIND-$PROFILE.lock"
  if [[ -f "$LOCK" ]]; then
    install "$TARGET" -r "$LOCK"
  elif [[ -f "$PROJECT_ROOT/requirements/$KIND-common.constraints" ]]; then
    install "$TARGET" -r "$PROJECT_ROOT/requirements/$KIND.txt" -c "$PROJECT_ROOT/requirements/$KIND-common.constraints"
  else
    install "$TARGET" -r "$PROJECT_ROOT/requirements/$KIND.txt"
  fi
  install "$TARGET" --no-deps -e "$PROJECT_ROOT"
  "$TARGET/bin/python" - "$PROFILE" "$KIND" <<'PY'
import sys,torch,torchvision,numpy,cv2,onnx,onnxruntime
if sys.argv[1]=='cuda':
    assert torch.cuda.is_available(), 'CUDA requested but unavailable; check the host driver'
    (torch.ones((16,16),device='cuda')@torch.ones((16,16),device='cuda')).sum().item()
else:
    (torch.ones((16,16))@torch.ones((16,16))).sum().item()
if sys.argv[2]=='train':
    import ultralytics
else:
    import xslim
print(sys.argv[2], 'ready:',torch.__version__, 'ONNX',onnx.__version__)
PY
  if command -v uv >/dev/null 2>&1; then
    uv pip check --python "$TARGET/bin/python"
    uv pip freeze --python "$TARGET/bin/python" > "$TARGET/resolved-requirements.txt"
  else
    "$TARGET/bin/python" -m pip check
    "$TARGET/bin/python" -m pip freeze > "$TARGET/resolved-requirements.txt"
  fi
done
"$PYTHON_BIN" - "$PROJECT_ROOT/.environments.json" "$ENV_ROOT" "$PROFILE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);root=Path(sys.argv[2]);profile=sys.argv[3]
data=json.loads(p.read_text()) if p.exists() else {}
data.update(profile=profile,train_python=str(root/f'train-{profile}'/'bin/python'),quant_python=str(root/f'quant-{profile}'/'bin/python'))
p.write_text(json.dumps(data,indent=2)+'\n')
PY
echo "Ready: $PROFILE. Copy config.example.yaml to config.yaml and run scripts/check_env.sh."
