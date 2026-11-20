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
exec /mnt/nvme1n1/sglang522/bin/sglang serve \
  --model-path /mnt/sda/models/Qwen3.8-27B-NVFP4 \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 8 \
  --speculative-ngram-match-type BFS \
  --speculative-ngram-max-trie-depth 8 \
  --speculative-ngram-max-bfs-breadth 4 \
  --enable-metrics \
  --tp-size 1  \
  --attention-backend flashinfer \
  --linear-attn-backend triton \
  --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 64 \
  --mem-fraction-static 0.84 \
  --context-length 131072 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --min-free-slots-delay 1 \
  --sampling-backend flashinfer \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000
