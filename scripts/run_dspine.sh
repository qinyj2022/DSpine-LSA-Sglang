#!/bin/bash
export PATH=/mnt/nvme1n1/sglang522/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=8
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TMPDIR=/mnt/sda/qyj1044/tmp
export SGLANG_CACHE_DIR=/mnt/sda/qyj1044/cache/sglang
export FLASHINFER_CACHE_DIR=/mnt/sda/qyj1044/cache/flashinfer
export TRITON_CACHE_DIR=/mnt/sda/qyj1044/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/sda/qyj1044/cache/inductor
export XDG_CACHE_HOME=/mnt/sda/qyj1044/cache/xdg
mkdir -p $TMPDIR $SGLANG_CACHE_DIR $FLASHINFER_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME

MODEL=/mnt/sda/models/Qwen3.8-27B-NVFP4
DRAFT=/mnt/sda/qyj1044/dspine_draft

exec /mnt/nvme1n1/sglang522/bin/sglang serve \
  --model-path "$MODEL" \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 \
  --attention-backend flashinfer \
  --speculative-draft-attention-backend flashinfer \
  --linear-attn-backend triton \
  --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 64 \
  --mem-fraction-static 0.78  \
  --context-length 131072 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --min-free-slots-delay 1 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000
