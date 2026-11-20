#!/bin/bash
# 把「verify CUDA graph」这轮优化固化成一个带校验和的 v1.1 发布快照。
# 用法: bash make_release_v11.sh
# 产物: /mnt/sda/qyj1044/releases/v1.1-lsa-dspine-graph/
#   只收本轮改动 + 启动脚本 + 补丁脚本；整包源码见 v1.0（v1.1 ⊃ v1.0，除下列文件外一致）
set -eu

REL=/mnt/sda/qyj1044/releases/v1.1-lsa-dspine-graph
V10=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
LSA=$SP/lsa/sglang
SRC=/mnt/sda/qyj1044
PY=/mnt/nvme1n1/sglang522/bin/python

rm -rf "$REL"
mkdir -p "$REL"/{models/qwen3_5,index,launch,patches}

# --- 本轮改动的两个源文件（相对 v1.0 的差异全在这里）---
cp "$LSA/models/qwen3_5/backend.py" "$REL/models/qwen3_5/backend.py"
cp "$LSA/index/selector.py"         "$REL/index/selector.py"
cp "$V10/models/qwen3_5/backend.py" "$REL/models/qwen3_5/backend.v10.py"
cp "$V10/index/registry.py"         "$REL/index/registry.v10.py"

# --- 启动脚本：graph 版为最优，nograph 为 A/B 基线 ---
for f in graph_lsa.sh nograph_lsa.sh graph_lsa_log.sh nograph_lsa_log.sh; do
    [ -f "$SRC/$f" ] && cp "$SRC/$f" "$REL/launch/$f"
done

# --- 本轮补丁脚本（幂等）---
for f in patch_lsa_spec_v4.py patch_lsa_spec_v5.py patch_lsa_spec_v7.py make_lsa_probe.py; do
    [ -f "$SRC/lsa_code/$f" ] && cp "$SRC/lsa_code/$f" "$REL/patches/$f"
done

STAMP=$(date -Iseconds)
cat > "$REL/VERSION" <<EOF
name: LSA + DSpine 联合运行 — verify CUDA graph 优化
version: v1.1
tag: v1.1-lsa-dspine-graph
created: $STAMP
status: 验收通过（图捕获成功 / accept 持平 / needle 命中 / @119k +15.6%）
parent: v1.0-lsa-dspine（整包源码与其余文件沿用 v1.0）

# 本轮相对 v1.0 的改动（三处，均在 LSA 包内）
  patch_lsa_spec_v4.py  backend.py select(): verify 改走 decode 的设备侧定形读取
                        （rows = req_pool_indices.repeat_interleave(k)，不写索引）
  patch_lsa_spec_v5.py  backend.py select(): 删除 verify 的 maintain 兜底
                        （LSA 只在两个元数据 hook 里维护，捕获 pass 无 rids）
  patch_lsa_spec_v7.py  selector.py: decode listing 缓冲按 request_count * draft_rows() 定尺
                        （TopK.listed 会用 held[:rows] 静默截断）
  + select_and_read(): 捕获拦截从 is_extend() 收窄为 is_extend_without_speculative()

# 启动
launch: launch/graph_lsa.sh   （= v1.0 配置去掉 --disable-cuda-graph）
serve:  bash tools/restart_any.sh graph_lsa.sh <日志名>
tunnel: ssh -N -L 127.0.0.1:30000:/mnt/sda/qyj1044/run/sglang.sock amd03

# 实测（同代码、同机）
  @119k decode       图关闭 81.35 tok/s (n=4) → 图开启 94.55 tok/s (n=5)   +15.6%
  短上下文五类负载    A51.1 B196.0 C79.5 D170.1 E111.9 tok/s（v1.0: A50.9 B178.8 C81.7 D162.1 E107.6）
  accept len         无图 5.157 / 有图 5.024（v1.0 的 v3 读法亦为 5.157 → 读法质量中立）
  needle 8k/32k/64k  全部命中
  每步耗时           短 27 ms → ≥16k 56 ms（与上下文无关）；LSA 单独 39.8 → 43.3 ms（+8.8%）
                      → 那 29 ms 跳变在 DSpine 侧，不在 LSA

# 已否决的方向（见 LSA_SPEC_HANDOFF.md §12）
  桶扫描去重      桶数 7× 变化步耗时不变
  top_k 调小      width 4612→1540 步耗时无改善
  cutedsl kernels 步耗时 +9%（回退）
  flashinfer 移植 读取非瓶颈，无 tok/s 收益
EOF

cat > "$REL/restore.sh" <<'EOF'
#!/bin/bash
# 把 v1.1 装回 site-packages（v1.1 的两个源文件 + v1.0 的 registry.py）。
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
LSA=$SP/lsa/sglang
for f in models/qwen3_5/backend.py index/selector.py index/registry.py; do
    [ -f "$LSA/$f" ] && cp "$LSA/$f" "$LSA/$f.before-restore" || true
done
cp "$HERE/models/qwen3_5/backend.py" "$LSA/models/qwen3_5/backend.py"
cp "$HERE/index/selector.py"         "$LSA/index/selector.py"
cp "$HERE/index/registry.v10.py"     "$LSA/index/registry.py"
/mnt/nvme1n1/sglang522/bin/python -m py_compile \
    "$LSA/models/qwen3_5/backend.py" "$LSA/index/selector.py" "$LSA/index/registry.py"
echo "restored v1.1 into $LSA"
grep -c 'PATCH spec v4' "$LSA/models/qwen3_5/backend.py" || true
grep -c 'PATCH spec v7' "$LSA/index/selector.py" || true
EOF
chmod +x "$REL/restore.sh"

cd "$REL"
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
echo "=== built: $REL ==="
cat VERSION
echo "=== MANIFEST ==="
cat MANIFEST.sha256
sha256sum -c MANIFEST.sha256 | grep -c ': OK$'
