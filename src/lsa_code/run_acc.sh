#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
export SGLANG_PLUGINS=lsa TMPDIR=/mnt/sda/qyj1044/tmp SGLANG_CACHE_DIR=/mnt/sda/qyj1044/cache/sglang FLASHINFER_CACHE_DIR=/mnt/sda/qyj1044/cache/flashinfer TRITON_CACHE_DIR=/mnt/sda/qyj1044/cache/triton
for M in lsa dense; do
  echo "########## $M ##########"
  /mnt/nvme1n1/sglang522/bin/python acctest.py $M 131072 2>&1 | grep -vE "server_args=|^\s*$" | tail -8
  sleep 8
done
echo "########## ACC DONE ##########"