#!/bin/bash
# 修正 v1.1 的两处：
#   1) v11-delta.diff 里 selector.py 的对比基线取上游（v1.0 发布时 selector.py 未被修改，
#      所以 v1.0 的快照里没有它，之前的 diff 直接报 No such file）
#   2) ORIGIN.md 说明 lsa-installed 里的 *.pre-v* / *.orig 是回滚备份，不是运行中的代码
# 然后重算校验和与 tarball。
set -eu

REL=/mnt/sda/qyj1044/releases/v1.1-lsa-dspine-graph
V10=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine

cd "$REL/source"

{
    echo "# v1.1 — 相对 v1.0 的改动（本轮 verify CUDA graph）"
    echo "# 生成: $(date -Iseconds)"
    echo "#"
    echo "# backend.py  左 = v1.0 快照里的已装副本，右 = v1.1 已装"
    echo "# selector.py 左 = 上游源码（v1.0 发布时 selector.py 未被修改，其快照里没有该文件），"
    echo "#             右 = v1.1 已装（本轮 v7）"
    echo
    echo "=================== models/qwen3_5/backend.py ==================="
    diff -u "$V10/models/qwen3_5/backend.py" "$REL/models/qwen3_5/backend.py" || true
    echo
    echo "=================== index/selector.py ==================="
    diff -u "lsa-upstream/src/lsa/sglang/index/selector.py" "lsa-installed/sglang/index/selector.py" || true
    echo
} > v11-delta.diff
echo "v11-delta.diff: $(wc -l < v11-delta.diff) 行"

cat >> ORIGIN.md <<'EOF'

## 关于 lsa-installed 里的备份文件

`source/lsa-installed/` 内可见 `*.orig` 与 `*.pre-v*` 后缀的文件——它们是取用 site-packages 时
一并拷来的**回滚备份**（每个补丁脚本执行前自动生成），**不是运行中的代码**：

| 备份 | 含义 |
|---|---|
| `backend.py.orig` | 上游原版，未打任何补丁 |
| `backend.py.pre-v2` / `.pre-v3` | 打 v2 / v3 补丁前的状态 |
| `backend.py.pre-v4` / `.pre-v5` | 打 v4 / v5 补丁前的状态 |
| `selector.py.pre-v7` | 打 v7 补丁前的状态（= 上游） |

运行中的代码只有 `backend.py`、`selector.py`、`registry.py` 三个文件。
EOF
echo "ORIGIN.md $(wc -l < ORIGIN.md) 行"

echo "=== 重算校验和 ==="
( cd "$REL/source" && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
( cd "$REL" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
cd "$REL"; echo "  top    : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"
cd source; echo "  source : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"

echo "=== 重打 tarball ==="
cd /mnt/sda/qyj1044/releases
rm -f v1.1-lsa-dspine-graph.tar.gz
tar -czf v1.1-lsa-dspine-graph.tar.gz v1.1-lsa-dspine-graph
ls -lh v1.1-lsa-dspine-graph.tar.gz | awk '{print "  size", $5}'
sha256sum v1.1-lsa-dspine-graph.tar.gz
