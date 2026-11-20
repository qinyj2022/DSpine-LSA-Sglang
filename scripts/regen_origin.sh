#!/bin/bash
# 修正 source/ORIGIN.md 里重复的 pip 元数据块，然后重算校验和并重打 tarball。
# 用法: bash regen_origin.sh
set -eu

REL=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
PY=/mnt/nvme1n1/sglang522/bin/python

{
    echo "# 源码备份来源 — v1.0-lsa-dspine"
    echo
    echo "备份时间: 2026-11-18T11:41:28+08:00（pip 元数据段 2026-11-18 修正重复项）"
    echo
    echo "## 三棵树的来历"
    echo
    echo '| 快照内路径 | 原始路径 | 说明 |'
    echo '|---|---|---|'
    echo "| \`source/lsa-installed/\` | \`$SP/lsa\` | **运行时权威代码**：当前进程实际 import 的 LSA 包，含本次全部补丁 |"
    echo "| \`source/lsa-upstream/\` | \`/mnt/sda/qyj1044/lsa_code/latent-sparse-attention\` | LSA 项目源码树（src/tests/configs/docs），上游参考；**不含本次补丁** |"
    echo "| \`source/dspine/\` | \`/mnt/sda/qyj1044/dspine_code/DSpine-main\` | DSpine 插件源码，以 \`pip install -e\` 可编辑安装挂进环境 |"
    echo
    echo "> 注意：补丁只存在于 \`lsa-installed/\`，因为它改的是 site-packages 里的已安装包；"
    echo "> \`lsa-upstream/\` 是干净的上游源码，\`diff -r\` 可直接看出本次改了什么（固化为 \`patches.diff\`）。"
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
    echo "改动量：backend.py 114 行、registry.py 7 行（见 patches.diff）。"
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
    echo "[editable 挂载（site-packages 内）]"
    ls -1 "$SP" | grep -iE 'dspine|specforge' || echo "  （无）"
    echo '```'
    echo
    echo "## 未备份的内容"
    echo
    echo "- \`__pycache__\` / \`*.pyc\`：可再生"
    echo "- 模型权重与 KV 缓存、LSA 蒸馏 map（体积大，路径见 ../VERSION）"
    echo "- DSpine 草稿权重（\`/mnt/sda/qyj1044/dspine_draft\`）"
} > "$REL/source/ORIGIN.md"

echo "ORIGIN.md 已重写（$(wc -l < "$REL/source/ORIGIN.md") 行）"
echo
bash /mnt/sda/qyj1044/make_patch_diff.sh
