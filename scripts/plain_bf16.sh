#!/bin/bash
export PATH=/mnt/nvme1n1/sglang522/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/mnt/sda/qyj1044/tmp
export SGLANG_CACHE_DIR=/mnt/sda/qyj1044/cache/sglang
export FLASHINFER_CACHE_DIR=/mnt/sda/qyj1044/cache/flashinfer
exec python -m sglang.launch_server --model-path /mnt/sda/qyj1044/dspine_train/model_view --attention-backend flashinfer --dtype bfloat16 --mem-fraction-static 0.80 --context-length 8192 --max-running-requests 4 --return-hidden-states-mode full --trust-remote-code --host 127.0.0.1 --port 30000