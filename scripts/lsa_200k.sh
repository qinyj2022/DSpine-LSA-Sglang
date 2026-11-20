#!/bin/bash
export PATH=/mnt/nvme1n1/sglang522/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TMPDIR=/mnt/sda/qyj1044/tmp
export SGLANG_CACHE_DIR=/mnt/sda/qyj1044/cache/sglang
export FLASHINFER_CACHE_DIR=/mnt/sda/qyj1044/cache/flashinfer
export TRITON_CACHE_DIR=/mnt/sda/qyj1044/cache/triton
export SGLANG_PLUGINS=lsa
MAP=/mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt
exec /mnt/nvme1n1/sglang522/bin/python /mnt/sda/qyj1044/lsa_code/sglang_uds.py \
  --model-path /mnt/sda/models/Qwen3.8-27B \
  --tp-size 1 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --page-size 1 \
  --disable-prefill-cuda-graph \
  --mem-fraction-static 0.85 \
  --context-length 262144 \
  --max-running-requests 8 \
  --trust-remote-code \
  --json-model-override-args "{\"lsa\": {\"projections\": \"$MAP\", \"query_heads\": \"all\"}}" \
  --decode-log-interval 1 --host 0.0.0.0 --port 30000