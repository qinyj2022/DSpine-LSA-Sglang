#!/bin/bash
cd /mnt/sda/qyj1044/datasets/perfectblend
BASE="https://hf-mirror.com/datasets/mlabonne/open-perfectblend/resolve/main/data"
for i in 0 1 2 3 4 5; do
  F="train-0000${i}-of-00006.parquet"
  [ -f "$F" ] && { echo "skip $F"; continue; }
  echo "--- $F $(date +%T) ---"
  curl -L --retry 8 --retry-delay 3 --connect-timeout 30 -C - -o "$F" "$BASE/$F" || echo "FAIL $F"
done
echo "PB_DL_DONE"
