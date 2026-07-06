#!/bin/bash
# 统一推理: 支持 configs.py 中注册的所有 dinov3+m2f 网络。
# 用法:
#   bash infer.sh pastis            # PASTIS 时序
#   bash infer.sh agri              # 农业单帧
#   bash infer.sh pastis --limit 40 # 只跑 40 个样本 (快速验证)
#   GPU=1 bash infer.sh agri        # 指定 GPU
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NET="${1:-pastis}"; shift || true
python infer.py --net "$NET" "$@"
