#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
export SGLANG_PLUGINS=lsa TMPDIR=/mnt/sda/qyj1044/tmp SGLANG_CACHE_DIR=/mnt/sda/qyj1044/cache/sglang FLASHINFER_CACHE_DIR=/mnt/sda/qyj1044/cache/flashinfer TRITON_CACHE_DIR=/mnt/sda/qyj1044/cache/triton
echo "########## LSA ##########"
/mnt/nvme1n1/sglang522/bin/python engtest.py lsa 131072 2>&1 | grep -vE "server_args=|^\s*$" | tail -20
sleep 10
echo "########## DENSE ##########"
/mnt/nvme1n1/sglang522/bin/python engtest.py dense 131072 2>&1 | grep -vE "server_args=|^\s*$" | tail -20
echo "########## ALL DONE ##########"