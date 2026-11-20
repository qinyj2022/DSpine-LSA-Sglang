#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="12.0"
cd /mnt/sda/qyj1044/fa2_check/flash_attn-2.8.3.post2
echo "=== FLASH_ATTN build start $(date +%T) arch=$TORCH_CUDA_ARCH_LIST ==="
/mnt/nvme1n1/sglang522/bin/python setup.py install 2>&1 | tail -40
echo "=== BUILD_EXIT=$? ==="
