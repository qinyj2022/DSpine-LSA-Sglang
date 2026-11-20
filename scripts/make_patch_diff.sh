#!/bin/bash
# 在 v1.0 发布文件夹里生成「相对上游 LSA 源码的补丁 diff」，并重算校验和。
# 用法: bash make_patch_diff.sh
set -eu

REL=/mnt/sda/qyj1044/releases/v1.0-lsa-dspine
cd "$REL/source"

{
    echo "# v1.0-lsa-dspine — 相对上游 LSA 源码的补丁"
    echo "#"
    echo "# 上游:  lsa-upstream/src/lsa/<文件>"
    echo "# 已部署: lsa-installed/<文件>   （= 当前 site-packages 里实际运行的代码）"
    echo "# 生成:  $(date -Iseconds)"
    echo "#"
    echo "# 应用方式见 ../restore.sh；重放需先回到上游版本再套补丁脚本 ../patches/patch_lsa_spec_v*.py"
    echo
    for f in sglang/models/qwen3_5/backend.py sglang/index/registry.py; do
        echo "=================== $f ==================="
        diff -u "lsa-upstream/src/lsa/$f" "lsa-installed/$f" || true
        echo
    done
} > patches.diff

echo "patches.diff: $(wc -l < patches.diff) 行"
grep -c '^[+-][^+-]' patches.diff || true

echo "=== 重算 source/MANIFEST.sha256 ==="
( cd "$REL/source" && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
echo "=== 重算顶层 MANIFEST.sha256 ==="
( cd "$REL" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )

echo "=== 校验 ==="
cd "$REL"
echo "  top  : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"
cd source
echo "  source: $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"

echo "=== 打包供本机镜像 ==="
cd /mnt/sda/qyj1044/releases
tar -czf v1.0-lsa-dspine.tar.gz v1.0-lsa-dspine
ls -lh v1.0-lsa-dspine.tar.gz | awk '{print "  " $5, $9}'
sha256sum v1.0-lsa-dspine.tar.gz
