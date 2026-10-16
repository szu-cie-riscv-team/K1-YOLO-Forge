#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${VENV_PATH:+$VENV_PATH/bin/python}"
if [[ -z "$PYTHON_BIN" && -f "$PROJECT_ROOT/.environments.json" ]]; then
  PYTHON_BIN="$(python3 - "$PROJECT_ROOT/.environments.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('inspect_python',''))
PY
)"
fi
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venvs/inspect/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Run bash scripts/setup_env.sh first, or set VENV_PATH." >&2
  exit 1
fi
exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/inspect_models.py" "$@"
