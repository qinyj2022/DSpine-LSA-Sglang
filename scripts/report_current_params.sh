#!/bin/bash
# 采集当前部署的显存占用与全部运行参数。
set -u
cd /mnt/sda/qyj1044
LOG=$(ls -t logs/*.log | head -1)

echo "########## 1. GPU 显存 ##########"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
           --format=csv,noheader
echo "--- 按进程 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

echo
echo "########## 2. 运行进程与全部参数 ##########"
pid=$(pgrep -f sglang_uds | head -1)
echo "pid=$pid   启动脚本=$(ps -eo cmd | grep '[s]glang_uds' | grep -oE '[a-z0-9_]+\.sh' | head -1)"
ps -eo cmd | grep '[s]glang_uds' | tr ' ' '\n' | grep -E '^--' | sed 's/^/  /' | head -70

echo
echo "########## 3. 服务端解析后的关键量 ##########"
grep -a -E 'max_total_num_tokens|max_running_requests is capped' "$LOG" | grep -v server_args | tail -2 | cut -c1-200
grep -a 'KV Cache is allocated' "$LOG" | tail -1 | cut -c1-160
grep -a 'lsa: memory' "$LOG" | tail -1 | cut -c1-220
grep -a 'Load weight end' "$LOG" | tail -1 | cut -c1-120
grep -a 'Memory baseline taken' "$LOG" | tail -1 | cut -c1-90

echo
echo "########## 4. LSA / 投机组件状态 ##########"
grep -a 'lsa: enabled' "$LOG" | head -1 | cut -c1-160
grep -a 'DSPINE ready' "$LOG" | tail -1 | cut -c1-200
grep -a 'Capture target verify CUDA graph end' "$LOG" | tail -1 | cut -c1-110
grep -a 'Linear attention kernel backend\|GDN kernel dispatcher' "$LOG" | tail -2 | cut -c1-130

echo
echo "########## 5. 日志健康度 ##########"
echo "  ready=$(grep -ac 'ready to roll' "$LOG")  SIGQUIT=$(grep -ac 'SIGQUIT' "$LOG")  RuntimeError=$(grep -ac 'RuntimeError' "$LOG")  fallback=$(grep -ac 'dspine-stable' "$LOG")"
echo "  服务已运行: $(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
