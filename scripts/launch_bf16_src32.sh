#!/bin/bash
cd /mnt/sda/qyj1044
pkill -9 -f "launch_server" 2>/dev/null; pkill -9 -f "engtest" 2>/dev/null; sleep 6
sed -e 's#--max-running-requests 8#--max-running-requests 32#' verify_bf16_src.sh > verify_bf16_src32.sh
chmod +x verify_bf16_src32.sh
sed -i 's/--host 0.0.0.0 --port 30000/--host 0.0.0.0 --port 30000/' verify_bf16_src32.sh
grep -nE "host|port|draft-model-path|max-running" verify_bf16_src32.sh | tail -5
truncate -s 0 logs/bf16_src32.log
setsid nohup bash /mnt/sda/qyj1044/verify_bf16_src32.sh > /mnt/sda/qyj1044/logs/bf16_src32.log 2>&1 < /dev/null &
echo LAUNCHED