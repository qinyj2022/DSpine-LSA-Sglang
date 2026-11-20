#!/bin/bash
# 用指定的启动脚本重启服务（UDS 监听 + TCP 桥接）。
# 用法: bash /mnt/sda/qyj1044/restart_any.sh <启动脚本名> <日志名>
# 注意: pkill 用 "[i]pc_bridge" 写法，避免把执行本脚本的 shell 自己也匹配掉。
set -u
LAUNCH="${1:?launch script required}"
LOG="${2:?log name required}"
cd /mnt/sda/qyj1044 || exit 1
pkill -9 -f '[i]pc_bridge'
pkill -9 -f '[s]glang_uds'
sleep 6
rm -f run/sglang.sock
setsid nohup /mnt/nvme1n1/sglang522/bin/python /mnt/sda/qyj1044/lsa_code/ipc_bridge.py \
    0.0.0.0 30000 /mnt/sda/qyj1044/run/sglang.sock > logs/bridge.log 2>&1 &
sleep 3
SGLANG_UDS_PATH=/mnt/sda/qyj1044/run/sglang.sock \
  setsid nohup bash "/mnt/sda/qyj1044/$LAUNCH" > "logs/$LOG" 2>&1 &
echo "restarted: $LAUNCH -> logs/$LOG"
