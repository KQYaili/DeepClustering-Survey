#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export DATA_ROOT="${DATA_ROOT:-$HOME/.cache/dc-survey/data}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONPATH="/home/lgd/venvs/dc-extra:${ROOT}/src:${PYTHONPATH:-}"
PY="${DC_PYTHON:-/home/lgd/anaconda3/bin/python}"
"$PY" tests/test_core.py
"$PY" scripts/run_queue.py "$@"
