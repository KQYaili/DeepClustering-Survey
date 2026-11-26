#!/usr/bin/env bash
# Reuse the existing Anaconda CUDA build. Do not let pip replace torch.
set -euo pipefail
PY=/home/lgd/anaconda3/bin/python
EXTRA=/home/lgd/venvs/dc-extra
mkdir -p "$EXTRA"
"$PY" -m pip install --target "$EXTRA" umap-learn
"$PY" -m pip install --target "$EXTRA" --no-deps open-clip-torch
"$PY" -m pip install --target "$EXTRA" ftfy regex huggingface_hub safetensors timm
PYTHONPATH="$EXTRA" "$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul_ok", float(y.mean()))
import umap
import open_clip
print("extras_ok")
PY
echo SETUP_OK
