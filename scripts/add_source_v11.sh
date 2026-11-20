#!/bin/bash
# 给 v1.1 快照补上完整源码备份（DSpine 全量 + LSA 已装包 + LSA 上游树），
# 并生成 ORIGIN.md / 两份 diff / 校验和 / tarball。可重复执行（幂等）。
# 用法: bash add_source_v11.sh
set -eu

REL=/mnt/sda/qyj1044/releases/v1.1-lsa-dspine-graph
V10=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
PY=/mnt/nvme1n1/sglang522/bin/python
SRC_LSA_INSTALLED=$SP/lsa
SRC_LSA_UPSTREAM=/mnt/sda/qyj1044/lsa_code/latent-sparse-attention
SRC_DSPINE=/mnt/sda/qyj1044/dspine_code/DSpine-main

copy_tree() {
    local src=$1 dst=$2
    [ -d "$src" ] || { echo "SKIP (missing): $src"; return 0; }
    rm -rf "$dst"; mkdir -p "$dst"
    ( cd "$src" && tar -cf - \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='.pytest_cache' . ) \
      | ( cd "$dst" && tar -xf - )
    printf '  %-24s %5s files  %6s\n' "$(basename "$dst")" \
        "$(find "$dst" -type f | wc -l)" "$(du -sh "$dst" | cut -f1)"
}

echo "=== 1. 拷贝源码树 ==="
mkdir -p "$REL/source"
copy_tree "$SRC_LSA_INSTALLED" "$REL/source/lsa-installed"
copy_tree "$SRC_LSA_UPSTREAM"  "$REL/source/lsa-upstream"
copy_tree "$SRC_DSPINE"        "$REL/source/dspine"

echo
echo "=== 2. ORIGIN.md ==="
{
    echo "# 源码备份来源 — v1.1-lsa-dspine-graph"
    echo
    echo "备份时间: $(date -Iseconds)"
    echo
    echo '| 快照内路径 | 原始路径 | 说明 |'
    echo '|---|---|---|'
    echo "| \`source/lsa-installed/\` | \`$SRC_LSA_INSTALLED\` | **运行时权威代码**：含 v1.0 补丁 + 本轮 v4/v5/v7 |"
    echo "| \`source/lsa-upstream/\` | \`$SRC_LSA_UPSTREAM\` | LSA 项目源码树，**干净上游**，不含任何补丁 |"
    echo "| \`source/dspine/\` | \`$SRC_DSPINE\` | DSpine 插件源码（\`pip install -e\` 可编辑安装） |"
    echo
    echo "## 两份 diff"
    echo
    echo "- \`patches.diff\` —— 已装 vs **上游**：累计两个版本的全部改动"
    echo "- \`v11-delta.diff\` —— 已装 vs **v1.0 已装**：只含本轮（verify CUDA graph）的改动"
    echo
    echo "## pip 元数据"
    echo
    echo '```'
    for pkg in lsa sglang-dspine; do
        echo "[$pkg]"
        "$PY" -m pip show "$pkg" 2>/dev/null \
            | grep -E '^(Name|Version|Location|Editable project location):' || echo "  (未安装)"
        echo
    done
    echo '```'
    echo
    echo "## 未备份"
    echo
    echo "- \`__pycache__\` / \`*.pyc\`（可再生）"
    echo "- 模型权重、KV 缓存、LSA 蒸馏 map、DSpine 草稿权重（路径见 \`../VERSION\`）"
} > "$REL/source/ORIGIN.md"
echo "  ORIGIN.md $(wc -l < "$REL/source/ORIGIN.md") 行"

echo
echo "=== 3. patches.diff（已装 vs 上游）==="
cd "$REL/source"
{
    echo "# v1.1 — 相对上游 LSA 源码的累计补丁"
    echo "# 上游: lsa-upstream/src/lsa/<文件>   已部署: lsa-installed/<文件>"
    echo "# 生成: $(date -Iseconds)"
    echo
    for f in sglang/models/qwen3_5/backend.py sglang/index/registry.py sglang/index/selector.py; do
        up="lsa-upstream/src/lsa/$f"
        [ -f "$up" ] || { echo "=== $f: 上游无此文件 ==="; continue; }
        echo "=================== $f ==================="
        diff -u "$up" "lsa-installed/$f" || true
        echo
    done
} > patches.diff
echo "  $(wc -l < patches.diff) 行"

echo
echo "=== 4. v11-delta.diff（本轮改动）==="
{
    echo "# v1.1 — 相对 v1.0 的改动（本轮 verify CUDA graph）"
    echo "# 左: v1.0 已装   右: v1.1 已装"
    echo "# 生成: $(date -Iseconds)"
    echo
    for f in models/qwen3_5/backend.py index/selector.py; do
        echo "=================== $f ==================="
        diff -u "$V10/$f" "$REL/$f" || true
        echo
    done
} > v11-delta.diff
echo "  $(wc -l < v11-delta.diff) 行"

echo
echo "=== 5. 校验和 ==="
( cd "$REL/source" && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
( cd "$REL" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
cd "$REL"
echo "  top    : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"
cd source
echo "  source : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"

echo
echo "=== 6. tarball ==="
cd /mnt/sda/qyj1044/releases
rm -f v1.1-lsa-dspine-graph.tar.gz
tar -czf v1.1-lsa-dspine-graph.tar.gz v1.1-lsa-dspine-graph
ls -lh v1.1-lsa-dspine-graph.tar.gz | awk '{print "  size", $5}'
sha256sum v1.1-lsa-dspine-graph.tar.gz

echo
echo "=== 7. 结构 ==="
cd "$REL"
printf '  %-34s %5s files  %6s\n' "v1.1/" "$(find . -type f | wc -l)" "$(du -sh . | cut -f1)"
for d in models index launch patches source source/lsa-installed source/lsa-upstream source/dspine; do
    [ -e "$d" ] && printf '  %-34s %5s files  %6s\n' "$d/" "$(find "$d" -type f | wc -l)" "$(du -sh "$d" | cut -f1)"
done
