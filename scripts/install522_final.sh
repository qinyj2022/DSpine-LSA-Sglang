#!/bin/bash
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
export PIP_DEFAULT_TIMEOUT=180
export TMPDIR=/mnt/sda/qyj1044/tmp
mkdir -p $TMPDIR
PY=/mnt/nvme1n1/sglang522/bin/python
W=/mnt/sda/qyj1044/dspine_wheels
for i in 1 2 3 4 5 6; do
  echo "=== attempt $i $(date +%T) ==="
  $PY -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com --timeout 180 --retries 20 \
    $W/flashinfer_python-0.7.1-py3-none-any.whl \
    $W/sglang_kernel-0.4.9-cp310-abi3-manylinux2014_x86_64.whl \
    "sglang==0.5.22" 2>&1 | tail -8
  rc=${PIPESTATUS[0]}
  echo "rc=$rc"
  [ "$rc" = "0" ] && { echo INSTALL522_OK; break; }
  sleep 8
done
