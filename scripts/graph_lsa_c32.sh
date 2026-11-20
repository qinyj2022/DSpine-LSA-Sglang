#!/bin/bash
export PATH=/mnt/nvme1n1/sglang522/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=8
export DSPINE_DRAFT_GRAPH=off
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TMPDIR=/mnt/sda/qyj1044/tmp
export SGLANG_CACHE_DIR=/mnt/sda/qyj1044/cache/sglang
export FLASHINFER_CACHE_DIR=/mnt/sda/qyj1044/cache/flashinfer
export TRITON_CACHE_DIR=/mnt/sda/qyj1044/cache/triton
mkdir -p $TMPDIR $SGLANG_CACHE_DIR $FLASHINFER_CACHE_DIR $TRITON_CACHE_DIR
exec /mnt/nvme1n1/sglang522/bin/python /mnt/sda/qyj1044/lsa_code/sglang_uds.py \
  --model-path /mnt/sda/qyj1044/dspine_train/model_view \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path /mnt/sda/qyj1044/dspine_draft \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 --dtype bfloat16 \
  --attention-backend triton \
  --page-size 1 \
  --disable-prefill-cuda-graph \
  --speculative-draft-attention-backend flashinfer \
  --linear-attn-backend triton \
  --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 160 \
  --mem-fraction-static 0.82 \
  --context-length 131072 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --min-free-slots-delay 1 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --json-model-override-args "{\"lsa\": {\"projections\": \"/mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt\", \"query_heads\": \"all\"}}" \
  --host 0.0.0.0 --port 30000
