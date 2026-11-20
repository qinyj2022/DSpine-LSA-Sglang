#!/bin/bash
# 把 DSpine 与 LSA 的完整插件代码备份进 v1.0 发布文件夹。
# 用法: bash add_source_backup.sh
#
# 备份三棵树（都排除 __pycache__/*.pyc/.git，保留 egg-info 因为它是可编辑安装的元数据）：
#   source/lsa-installed/  当前真正在跑的 LSA 包（含本次补丁）—— 运行时权威代码
#   source/lsa-upstream/   LSA 项目源码树（上游参考 + 测试 + 配置）
#   source/dspine/         DSpine 插件源码（pip install -e 的可编辑安装源）
#
# 产物：source/ORIGIN.md、source/MANIFEST.sha256，并重算顶层 MANIFEST.sha256。
set -eu

REL=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
SRC_LSA_INSTALLED=$SP/lsa
SRC_LSA_UPSTREAM=/mnt/sda/qyj1044/lsa_code/latent-sparse-attention
SRC_DSPINE=/mnt/sda/qyj1044/dspine_code/DSpine-main
PY=/mnt/nvme1n1/sglang522/bin/python

copy_tree() {
    local src=$1 dst=$2
    [ -d "$src" ] || { echo "SKIP (missing): $src"; return 0; }
    rm -rf "$dst"
    mkdir -p "$dst"
    ( cd "$src" && tar -cf - \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='.pytest_cache' . ) \
      | ( cd "$dst" && tar -xf - )
    printf '  %-28s %6s files  %6s\n' "$(basename "$dst")" \
        "$(find "$dst" -type f | wc -l)" "$(du -sh "$dst" | cut -f1)"
}

echo "=== 拷贝源码树 ==="
mkdir -p "$REL/source"
copy_tree "$SRC_LSA_INSTALLED" "$REL/source/lsa-installed"
copy_tree "$SRC_LSA_UPSTREAM"  "$REL/source/lsa-upstream"
copy_tree "$SRC_DSPINE"        "$REL/source/dspine"

echo
echo "=== 记录来源与安装方式 ==="
{
    echo "# 源码备份来源 — v1.0-lsa-dspine"
    echo
    echo "备份时间: $(date -Iseconds)"
    echo
    echo "## 三棵树的来历"
    echo
    echo '| 快照内路径 | 原始路径 | 说明 |'
    echo '|---|---|---|'
    echo "| \`source/lsa-installed/\` | \`$SRC_LSA_INSTALLED\` | **运行时权威代码**：当前进程实际 import 的 LSA 包，含本次全部补丁 |"
    echo "| \`source/lsa-upstream/\` | \`$SRC_LSA_UPSTREAM\` | LSA 项目源码树（src/tests/configs/docs），上游参考；**不含本次补丁** |"
    echo "| \`source/dspine/\` | \`$SRC_DSPINE\` | DSpine 插件源码，以 \`pip install -e\` 可编辑安装挂进环境 |"
    echo
    echo "> 注意：补丁只存在于 \`lsa-installed/\`，因为它改的是 site-packages 里的已安装包；"
    echo "> \`lsa-upstream/\` 是干净的上游源码，用 \`diff -r\` 可直接看出本次改了什么。"
    echo
    echo "## 与上游的差异（即本次补丁）"
    echo
    echo '```'
    echo "  backend.py  [PATCH spec]     解除投机拒绝 + verify 走 extend 形态 + maintain 补偿"
    echo "              [PATCH spec v2]  dbg()/cache_ends() 诊断；每请求 token 数取自 out_cache_loc"
    echo "              [PATCH spec v3]  verify 改为只读选择，draft 块不写入索引（核心修复）"
    echo "              [LSA-DBG]        key_and_query 失败时的一次性诊断"
    echo "  registry.py [PATCH spec]      ends_of() 为 verify 合成 lengths"
    echo '```'
    echo
    echo "## pip 元数据"
    echo
    echo '```'
    for pkg in lsa sglang-dspine sglang_dspine; do
        out=$("$PY" -m pip show "$pkg" 2>/dev/null | grep -E '^(Name|Version|Location|Editable project location):' || true)
        [ -n "$out" ] && { echo "[$pkg]"; echo "$out"; echo; }
    done
    echo "[editable 挂载]"
    ls -1 "$SP" | grep -iE 'dspine|__editable__.*dspine' || echo "  (site-packages 内无 dspine* 目录，走 editable finder)"
    find "$SP" -maxdepth 1 -name '*dspine*' -o -maxdepth 1 -name '*.pth' 2>/dev/null | head -5
    echo '```'
    echo
    echo "## 未备份的内容"
    echo
    echo "- \`__pycache__\` / \`*.pyc\`：可再生"
    echo "- 模型权重与 KV 缓存、LSA 蒸馏 map（体积大，路径见 VERSION）"
    echo "- DSpine 草稿权重（\`/mnt/sda/qyj1044/dspine_draft\`）"
} > "$REL/source/ORIGIN.md"

echo
echo "=== 生成 source/MANIFEST.sha256 ==="
( cd "$REL/source" && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
echo "  $(wc -l < "$REL/source/MANIFEST.sha256") 个文件已校验"

echo
echo "=== 重算顶层 MANIFEST.sha256（含 source/）==="
( cd "$REL" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
echo "  $(wc -l < "$REL/MANIFEST.sha256") 个文件已校验"

echo
echo "=== 最终结构 ==="
cd "$REL"
printf '  %-52s %6s files  %6s\n' "releases/v1.0-lsa-dspine/" "$(find . -type f | wc -l)" "$(du -sh . | cut -f1)"
for d in models index launch patches tools source source/lsa-installed source/lsa-upstream source/dspine; do
    [ -e "$d" ] && printf '  %-52s %6s files  %6s\n' "$d/" "$(find "$d" -type f | wc -l)" "$(du -sh "$d" | cut -f1)"
done
