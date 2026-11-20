#!/bin/bash
# 把当前跑通的 LSA + DSpine 状态固化成一个带校验和的发布快照。
# 用法: bash make_release_v1.sh
# 产物: /mnt/sda/qyj1044/releases/v1.0-lsa-dspine/
#   VERSION          版本元数据（含实测数据与环境）
#   MANIFEST.sha256  目录内所有文件的校验和
#   models/..., index/..., launch/..., patches/..., tools/
#   restore.sh       把快照装回 site-packages，用于回滚或重新部署
set -eu

REL=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
LSA=$SP/lsa/sglang
SRC=/mnt/sda/qyj1044

rm -rf "$REL"
mkdir -p "$REL"/{models/qwen3_5,index,launch,patches,tools}

# --- 被修补过的 LSA 源文件（这是 V1.0 的核心内容）---
cp "$LSA/models/qwen3_5/backend.py" "$REL/models/qwen3_5/backend.py"
cp "$LSA/index/registry.py"         "$REL/index/registry.py"

# 同时留一份原始版本，方便比较差异
cp "$LSA/models/qwen3_5/backend.py.orig" "$REL/models/qwen3_5/backend.py.upstream"
cp "$LSA/index/registry.py.orig"         "$REL/index/registry.py.upstream"

# --- 启动脚本 ---
for f in dspine_lsa_v3.sh bf16_src_uds.sh abl_lsa.sh abl_dspine_only.sh; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$REL/launch/$f"
done

# --- 补丁脚本（可重放，幂等）---
for f in patch_lsa_spec_v2.py patch_lsa_spec_v3.py; do
  [ -f "$SRC/lsa_code/$f" ] && cp "$SRC/lsa_code/$f" "$REL/patches/$f"
done

# --- 运维脚本 ---
for f in restart_any.sh accept_stats.sh; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$REL/tools/$f"
done

# --- 版本元数据 ---
SGLANG_VER=$(/mnt/nvme1n1/sglang522/bin/python -c 'import sglang; print(sglang.__version__)')
TORCH_VER=$(/mnt/nvme1n1/sglang522/bin/python -c 'import torch; print(torch.__version__)')
LSA_VER=$(/mnt/nvme1n1/sglang522/bin/python -c 'import lsa; print(getattr(lsa, "__version__", "unknown"))')
KERNEL=$(uname -r)
STAMP=$(date -Iseconds)

cat > "$REL/VERSION" <<EOF
name: LSA + DSpine 联合运行
version: v1.0
tag: v1.0-lsa-dspine
created: $STAMP
status: 跑通并验证（verify 行数对齐 / 长上下文 needle 命中 / 投机接受率持平）

# 运行环境
host: $(hostname)
kernel: $KERNEL
python: /mnt/nvme1n1/sglang522/bin/python
sglang: $SGLANG_VER
torch: $TORCH_VER
lsa: $LSA_VER
gpu: sm120 (RTX PRO 6000 Blackwell Max-Q, 97887 MiB)

# 关键资产
model: /mnt/sda/qyj1044/dspine_train/model_view
draft: /mnt/sda/qyj1044/dspine_draft
lsa_map: /mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt

# 启动方式
launch: launch/dspine_lsa_v3.sh
serve: bash tools/restart_any.sh dspine_lsa_v3.sh <日志名>
tunnel: ssh -N -L 127.0.0.1:30000:/mnt/sda/qyj1044/run/sglang.sock amd03

# 补丁清单（backend.py / registry.py 内）
  [PATCH spec]     解除投机拒绝 + verify 走 extend 形态 + maintain 补偿 + ends_of 合成长度
  [PATCH spec v2]  dbg()/cache_ends() 诊断；verify 每请求 token 数取自 out_cache_loc
  [PATCH spec v3]  verify 改为只读选择，draft 块不写入索引（核心修复）
  [LSA-DBG]        key_and_query 失败时的一次性诊断（LSA_SPEC_DEBUG=1）

# 实测（同 prompt、temperature 0、单流）
  accept_len         DSpine-only 5.510  vs LSA+DSpine 5.157  (93.6%)
  needle 8k/32k/64k  全部命中
  decode @64k        90.48 vs 90.89 tok/s  (1.00x)
  decode @119k       63.94 vs 93.22 tok/s  (1.46x)
  prefill @119k      2865  vs 4032  tok/s  (1.41x)
  短上下文五类负载   LSA 为纯投机的 0.59-0.88x（64k 以下 LSA 净亏）
EOF

# --- 回滚/部署脚本 ---
cat > "$REL/restore.sh" <<'EOF'
#!/bin/bash
# 把本快照装回 site-packages（重新部署或从别的版本回滚到 v1.0）。
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
LSA=$SP/lsa/sglang
[ -f "$LSA/models/qwen3_5/backend.py" ] && cp "$LSA/models/qwen3_5/backend.py" "$LSA/models/qwen3_5/backend.py.before-restore" || true
cp "$HERE/models/qwen3_5/backend.py" "$LSA/models/qwen3_5/backend.py"
cp "$HERE/index/registry.py"         "$LSA/index/registry.py"
/mnt/nvme1n1/sglang522/bin/python -m py_compile "$LSA/models/qwen3_5/backend.py" "$LSA/index/registry.py"
echo "restored v1.0 into $LSA"
grep -c 'PATCH spec v3' "$LSA/models/qwen3_5/backend.py" || true
EOF
chmod +x "$REL/restore.sh"

# --- 校验和清单（最后生成，覆盖上面所有文件）---
cd "$REL"
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256

echo "=== release built: $REL ==="
cat VERSION
echo
echo "=== MANIFEST.sha256 ==="
cat MANIFEST.sha256
