#!/bin/bash
# 给 v1.1 快照内的 VERSION 补上源码备份段，然后重算校验和并重打 tarball。
set -eu

REL=/mnt/sda/qyj1044/releases/v1.1-lsa-dspine-graph
cd "$REL"

if ! grep -q '^SOURCE_LSA_INSTALLED:' VERSION; then
    cat >> VERSION <<'EOF'

# 完整源码备份（与 v1.0 同构，均排除 __pycache__/*.pyc/.git）
SOURCE_LSA_INSTALLED: source/lsa-installed/  353 files  5.8 MB  运行时权威代码（含 v1.0 补丁 + 本轮 v4/v5/v7）
SOURCE_LSA_UPSTREAM:  source/lsa-upstream/  1161 files   13 MB  干净上游源码树
SOURCE_DSPINE:        source/dspine/         140 files  2.6 MB  DSpine 插件源码（editable 安装源）
SOURCE_ORIGIN:  source/ORIGIN.md        来源路径 / pip 元数据 / 回滚备份说明 / 未备份内容
SOURCE_DIFFS:   source/patches.diff     已装 vs 上游，累计补丁（215 行）
                source/v11-delta.diff   已装 vs v1.0，仅本轮改动（166 行）
SOURCE_MANIFEST: source/MANIFEST.sha256 1656 项校验和
TOP_MANIFEST:    MANIFEST.sha256        1671 项校验和
EOF
    echo "VERSION 已补源码备份段：$(wc -l < VERSION) 行"
else
    echo "VERSION 已含源码备份段"
fi

echo "=== 重算校验和 ==="
( cd "$REL/source" && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
( cd "$REL" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
echo "  top    : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"
( cd source && echo "  source : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)" )

echo "=== 重打 tarball ==="
cd /mnt/sda/qyj1044/releases
rm -f v1.1-lsa-dspine-graph.tar.gz
tar -czf v1.1-lsa-dspine-graph.tar.gz v1.1-lsa-dspine-graph
ls -lh v1.1-lsa-dspine-graph.tar.gz | awk '{print "  size", $5}'
sha256sum v1.1-lsa-dspine-graph.tar.gz
