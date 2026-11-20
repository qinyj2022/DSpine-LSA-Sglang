#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="12.0"
cd /mnt/sda/qyj1044/fa2_check/flash_attn-2.8.3.post2
rm -rf build
/mnt/nvme1n1/sglang522/bin/python setup.py build 2>&1
