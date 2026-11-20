#!/bin/bash
# 重启 DSpine + LSA 服务（UDS 监听 + TCP 桥接）。
# 用法: bash /mnt/sda/qyj1044/restart_lsa.sh [日志名]
# 注意: 这里的 pkill 用 "[i]pc_bridge" 写法，避免把执行本脚本的 shell 自己也匹配掉。
set -u
LOG="${1:-lsa_v5.log}"
cd /mnt/sda/qyj1044 || exit 1
pkill -9 -f '[i]pc_bridge'
pkill -9 -f '[s]glang_uds'
sleep 6
rm -f run/sglang.sock
setsid nohup /mnt/nvme1n1/sglang522/bin/python /mnt/sda/qyj1044/lsa_code/ipc_bridge.py \
    0.0.0.0 30000 /mnt/sda/qyj1044/run/sglang.sock > logs/bridge.log 2>&1 &
sleep 3
LSA_SPEC_DEBUG=1 SGLANG_UDS_PATH=/mnt/sda/qyj1044/run/sglang.sock \
  setsid nohup bash /mnt/sda/qyj1044/dspine_lsa_v3.sh > "logs/$LOG" 2>&1 &
echo "restarted -> logs/$LOG"
