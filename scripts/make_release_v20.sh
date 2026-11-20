#!/bin/bash
# 把当前 Qwen3.8-27B 的 LSA + DSpine 部署固化成 v2.0 发布快照。
# 用法: bash make_release_v20.sh
#
# 收什么：
#   - 三处已打补丁的源文件（LSA 4 个 / sglang 1 个 / DSpine 1 个）+ 上游对照
#   - 全部启动脚本（含 FP4、MTP、并发档、SPL 包装）
#   - 全部补丁脚本（幂等，可重放）
#   - FP4「权重仍量化、KV 回 bf16」的 config.json 变体
#   - 完整源码备份（LSA 已装包 / LSA 上游树 / DSpine 源码）
#   - 三份 diff、ORIGIN.md、VERSION、restore.sh、校验和、tarball
set -eu

REL=/mnt/sda/qyj1044/releases/v2.0-qwen38-lsa-dspine
V11=/mnt/sda/qyj1044/releases/v1.1-lsa-dspine-graph
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
LSA=$SP/lsa/sglang
SGL=$SP/sglang/srt
SRC=/mnt/sda/qyj1044
DSP=$SRC/dspine_code/DSpine-main
PY=/mnt/nvme1n1/sglang522/bin/python

rm -rf "$REL"
mkdir -p "$REL"/{lsa,index,sglang,dspine,launch,patches,configs,diffs,source}

echo "=== 1. 已打补丁的源文件 ==="
cp "$LSA/models/qwen3_5/backend.py"           "$REL/lsa/backend.py"
cp "$LSA/index/registry.py"                   "$REL/index/registry.py"
cp "$LSA/index/selector.py"                   "$REL/index/selector.py"
cp "$LSA/index/layer_index.py"                "$REL/index/layer_index.py"
cp "$SGL/speculative/eagle_worker_common.py"  "$REL/sglang/eagle_worker_common.py"
cp "$DSP/sglang_dspine/plugin.py"             "$REL/dspine/plugin.py"
for pair in \
    "$LSA/models/qwen3_5/backend.py.orig:$REL/lsa/backend.upstream.py" \
    "$LSA/index/registry.py.orig:$REL/index/registry.upstream.py" \
    "$SGL/speculative/eagle_worker_common.py.pre-spl:$REL/sglang/eagle_worker_common.pre-patch.py" \
    "$DSP/sglang_dspine/plugin.py.pre-stable:$REL/dspine/plugin.pre-patch.py" ; do
    src=${pair%%:*}; dst=${pair##*:}
    [ -f "$src" ] && cp "$src" "$dst"
done
for f in "$REL"/lsa/*.py "$REL"/index/*.py "$REL"/sglang/*.py "$REL"/dspine/*.py; do
    printf '  %-42s %s 处 PATCH\n' "$(echo "$f" | sed "s#$REL/##")" "$(grep -c 'PATCH ' "$f" || true)"
done

echo "=== 2. 启动脚本 ==="
for f in graph_lsa.sh nograph_lsa.sh graph_lsa_log.sh nograph_lsa_log.sh \
         graph_lsa_mtp.sh graph_lsa_c19.sh graph_lsa_c32.sh \
         fp4_lsa_kvbf16.sh fp4_lsa_mtp_kvbf16.sh fp4_lsa_dspine_kvbf16.sh \
         fp4_prev_ar_uds.sh fp4_lsa_alone.sh fp4_dense_alone.sh \
         spl_mtp8_keep4.sh spl_mtp8_keep0.sh; do
    [ -f "$SRC/$f" ] && cp "$SRC/$f" "$REL/launch/$f"
done
ls -1 "$REL/launch" | sed 's/^/  /'

echo "=== 3. 补丁脚本 ==="
for f in "$SRC"/lsa_code/patch_lsa_spec_v*.py "$SRC"/patch_spl.py "$SRC"/patch_spl2.py "$SRC"/patch_dspine_stable.py; do
    [ -f "$f" ] && cp "$f" "$REL/patches/"
done
ls -1 "$REL/patches" | sed 's/^/  /'

echo "=== 4. FP4 KV-bf16 config 变体 ==="
if [ -f /mnt/sda/models/Qwen3.8-27B-NVFP4-kvbf16/config.json ]; then
    cp /mnt/sda/models/Qwen3.8-27B-NVFP4-kvbf16/config.json "$REL/configs/config.nvfp4-kvbf16.json"
    echo "  config.nvfp4-kvbf16.json  (摘掉 kv_cache_scheme，权重不变)"
else
    echo "  （未找到变体目录）"
fi

echo "=== 5. 完整源码备份 ==="
copy_tree() {
    local src=$1 dst=$2
    [ -d "$src" ] || { echo "  SKIP $src"; return 0; }
    rm -rf "$dst"; mkdir -p "$dst"
    ( cd "$src" && tar -cf - --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' . ) \
      | ( cd "$dst" && tar -xf - )
    printf '  %-24s %5s files  %6s\n' "$(basename "$dst")" \
        "$(find "$dst" -type f | wc -l)" "$(du -sh "$dst" | cut -f1)"
}
copy_tree "$SP/lsa"                       "$REL/source/lsa-installed"
copy_tree "$SRC/lsa_code/latent-sparse-attention" "$REL/source/lsa-upstream"
copy_tree "$DSP"                          "$REL/source/dspine"

echo "=== 6. 三份 diff ==="
( cd "$REL"
  { echo "# LSA 已装 vs 上游（累计补丁）"; echo
    for f in models/qwen3_5/backend.py index/registry.py index/selector.py index/layer_index.py; do
        up="source/lsa-upstream/src/lsa/sglang/$f"
        [ -f "$up" ] || continue
        echo "=================== $f ==================="
        diff -u "$up" "source/lsa-installed/sglang/$f" || true
        echo
    done
  } > diffs/lsa-spec.diff
  { echo "# sglang eagle_worker_common.py：本轮 SPL 串行草稿补丁"; echo
    diff -u sglang/eagle_worker_common.pre-patch.py sglang/eagle_worker_common.py || true
  } > diffs/sglang-spl.diff
  { echo "# DSpine plugin.py：采样轮缺链时的优雅降级补丁"; echo
    diff -u dspine/plugin.pre-patch.py dspine/plugin.py || true
  } > diffs/dspine-stable.diff
)
wc -l "$REL"/diffs/*.diff | sed 's/^/  /' | tail -4

STAMP=$(date -Iseconds)
cat > "$REL/VERSION" <<EOF
name: Qwen3.8-27B LSA + DSpine 部署
version: v2.0
tag: v2.0-qwen38-lsa-dspine
created: $STAMP
status: 生产可用（采样请求稳定、verify CUDA graph、并发 12/16/19 均验证）
parent: v1.1-lsa-dspine-graph

# 相对 v1.1 的改动
  LSA backend.py     [PATCH spec v4]  verify 走 decode 的设备侧定形读取（可捕获 CUDA graph）
                     [PATCH spec v5]  删除 verify 的 maintain 兜底（捕获 pass 无 rids）
                     [PATCH spec v10] 非 bf16 KV 缓存时按 k_scale 反量化读出（fp8 KV 支持）
                     [PATCH spec v11] 无 k_scale 时按 1.0 反量化
                     +               捕获拦截收窄为 is_extend_without_speculative()
  LSA registry.py    [PATCH spec v9]  replay 视图无 host seq_lens 时回落设备张量（NEXTN/MTP 兼容）
                     [PATCH spec v12] padding 批按真实 rids 截断（**并发 >1 的致命 bug**）
  LSA selector.py    [PATCH spec v7]  decode listing 缓冲按 request_count*draft_rows 定尺
  LSA layer_index.py [PATCH spec v10] 非 bf16 缓存跳过索引键回写（避免污染量化缓存）
  sglang eagle…common.py [PATCH spl/spl2] 草稿链尾部用 prompt lookup 续接（默认关闭，SPL_ENABLE=1 开启）
  DSpine plugin.py   [PATCH dspine-stable] 采样轮缺草稿链时降级到 stock DFlash 验证，不再杀 scheduler

# 启动
launch:  launch/graph_lsa.sh        （LSA + DSpine，bf16，verify 图开启）★推荐
variants: launch/graph_lsa_mtp.sh   （草稿换 MTP/NEXTN，采样路径更保守）
          launch/graph_lsa_c19.sh   （mamba 96 / 0.78 → 并发上限 19）
          launch/fp4_lsa_kvbf16.sh  （FP4 量化权重 + KV bf16 + LSA，无投机）
          launch/fp4_lsa_mtp_kvbf16.sh / fp4_lsa_dspine_kvbf16.sh
          launch/spl_mtp8_keep4.sh  （串行草稿实验，默认不改行为）
serve:   bash restart_any.sh <脚本> <日志名>
tunnel:  ssh -N -L 127.0.0.1:30000:/mnt/sda/qyj1044/run/sglang.sock amd03
         （每次重启远端服务后必须重建；用 127.0.0.1，勿用 localhost）

# 实测（单流短上下文五类负载 / 长上下文 / 并发）
  默认配置（graph_lsa.sh）:
    A 52.4  B 198.2  C 81.0  D 173.4  E 113.2 tok/s   均值 123.6   accept 5.02-5.36
    @119k decode 94.55 tok/s（无图 81.35 → 图开启，+15.6%）
    并发: 1→87.7  2→157.3  4→278.5  8→472.4  12→495.4 tok/s
  放宽并发（graph_lsa_c19.sh，KV 池 124850→20974）:
    并发 12→462.7  16→605.6  19→553.3 tok/s   （峰值 16 并发 ≈ 606）
  FP4 权重（权重 FP8/NVFP4 + KV bf16）:
    每步耗时 39.8→26.6 ms（1.5x），needle 8k/24k 命中
    但 FP4 + DSpine accept 5.02→1.37（草稿针对 bf16 目标训练），故不推荐
  MTP on FP4: accept 4.46，steps=8 最佳（均值 94.1）
  DSpine 稳定性: 62 次采样/长提示/并发请求零崩溃（修复前首次采样请求即崩）

# 已知限制
  - FP4 包必须摘掉 kv_cache_scheme 才能让 KV 回 bf16（否则 auto→fp8，LSA 拒或数值崩坏）
  - FP4 与 DSpine 组合 accept 崩塌，不推荐
  - 串行草稿（SPL）在当前负载集上与基线持平，仅在可复制文本上有潜力
  - 远端出向 TCP 100 KiB 上限：HTTP 必须走 UDS；torch TCPStore 仍走 TCP，可能偶发崩溃
EOF

cat > "$REL/restore.sh" <<'EOF'
#!/bin/bash
# 把 v2.0 的三处补丁装回 site-packages / DSpine 源码树。
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
LSA=$SP/lsa/sglang
SGL=$SP/sglang/srt/speculative
DSP=/mnt/sda/qyj1044/dspine_code/DSpine-main/sglang_dspine
for pair in "$LSA/models/qwen3_5/backend.py:$HERE/lsa/backend.py" \
            "$LSA/index/registry.py:$HERE/index/registry.py" \
            "$LSA/index/selector.py:$HERE/index/selector.py" \
            "$LSA/index/layer_index.py:$HERE/index/layer_index.py" \
            "$SGL/eagle_worker_common.py:$HERE/sglang/eagle_worker_common.py" \
            "$DSP/plugin.py:$HERE/dspine/plugin.py"; do
    dst=${pair%%:*}; src=${pair##*:}
    [ -f "$dst" ] && cp "$dst" "$dst.before-restore" || true
    cp "$src" "$dst"
    echo "  installed $dst"
done
/mnt/nvme1n1/sglang522/bin/python -m py_compile \
    "$LSA/models/qwen3_5/backend.py" "$LSA/index/registry.py" "$LSA/index/selector.py" \
    "$LSA/index/layer_index.py" "$SGL/eagle_worker_common.py" "$DSP/plugin.py"
echo "restored v2.0"
EOF
chmod +x "$REL/restore.sh"

echo "=== 7. ORIGIN.md ==="
{
    echo "# 源码备份来源 — v2.0-qwen38-lsa-dspine"
    echo
    echo "备份时间: $STAMP"
    echo
    echo '| 快照内路径 | 原始路径 | 说明 |'
    echo '|---|---|---|'
    echo "| \`source/lsa-installed/\` | \`$SP/lsa\` | 运行时权威代码（含 v4/v5/v7/v9/v10/v11/v12） |"
    echo "| \`source/lsa-upstream/\` | \`$SRC/lsa_code/latent-sparse-attention\` | 干净上游 LSA 源码树 |"
    echo "| \`source/dspine/\` | \`$DSP\` | DSpine 插件源码（editable 安装） |"
    echo "| \`sglang/eagle_worker_common.py\` | \`$SP/sglang/srt/speculative/eagle_worker_common.py\` | SPL 串行草稿补丁（sglang 侧，需单独安装） |"
    echo
    echo "## 回滚备份说明"
    echo
    echo "\`*.orig\` / \`.pre-*\` 是各补丁脚本执行前自动生成的回滚点，不是运行中的代码。"
    echo
    echo "## pip 元数据"
    echo
    echo '```'
    for pkg in lsa sglang-dspine; do
        echo "[$pkg]"
        "$PY" -m pip show "$pkg" 2>/dev/null | grep -E '^(Name|Version|Location|Editable project location):' || echo "  (未安装)"
        echo
    done
    echo '```'
} > "$REL/source/ORIGIN.md"

echo "=== 8. 校验和 ==="
( cd "$REL/source" && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
( cd "$REL" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
cd "$REL"
echo "  top    : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"
( cd source && echo "  source : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)" )

echo "=== 9. tarball ==="
cd /mnt/sda/qyj1044/releases
rm -f v2.0-qwen38-lsa-dspine.tar.gz
tar -czf v2.0-qwen38-lsa-dspine.tar.gz v2.0-qwen38-lsa-dspine
ls -lh v2.0-qwen38-lsa-dspine.tar.gz | awk '{print "  size", $5}'
sha256sum v2.0-qwen38-lsa-dspine.tar.gz

echo "=== 10. 结构 ==="
cd "$REL"
printf '  %-30s %5s files  %6s\n' "v2.0/" "$(find . -type f | wc -l)" "$(du -sh . | cut -f1)"
for d in lsa index sglang dspine launch patches configs diffs source; do
    [ -d "$d" ] && printf '  %-30s %5s files  %6s\n' "$d/" "$(find "$d" -type f | wc -l)" "$(du -sh "$d" | cut -f1)"
done
