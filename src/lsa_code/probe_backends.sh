#!/bin/bash
# 只读探针：确认这代栈里有没有 FA2、flashinfer 能做什么、LSA 的 flash 门在哪。
set -u
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
PY=/mnt/nvme1n1/sglang522/bin/python

echo "=== 1. FA2 引用（含上下文）==="
grep -n 'flash_attn_2\|FA2\|flashattn2' "$SP/sglang/srt/layers/attention/flashinfer_backend.py" | head -4
echo "--- qsa / qwen_sparse ---"
grep -n 'flash_attn_2\|FA2' "$SP/sglang/srt/layers/attention/qsa/kernel.py" | head -3

echo
echo "=== 2. flashinfer 能力 ==="
"$PY" - <<'PY'
import flashinfer
print("version:", flashinfer.__version__)
for name in (
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchMLAPagedAttentionWrapper",
):
    print(f"  {name}: {hasattr(flashinfer, name)}")
PY

echo
echo "=== 3. LSA 是否碰 flashinfer ==="
grep -rin flashinfer --include=*.py "$SP/lsa/sglang/" | head -3
echo "（以上为空表示 LSA 完全不碰 flashinfer）"

echo
echo "=== 4. LSA 的 flash 门 ==="
grep -n 'def flash_reads' -A 14 "$SP/lsa/sglang/models/qwen3_5/backend.py"

echo
echo "=== 5. LSA 的 Arch 枚举 ==="
grep -rn 'HOPPER\|BLACKWELL\|SM1' "$SP/lsa/core/kernels/constants.py" | head -10

echo
echo "=== 6. sglang 后端注册名 ==="
grep -rn 'AttentionBackendEnum' -A 30 "$SP/sglang/srt/layers/attention/attention_registry.py" 2>/dev/null | head -32
