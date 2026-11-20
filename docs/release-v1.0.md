# Release v1.0 — LSA + DSpine 联合运行

| 项 | 值 |
|---|---|
| 版本 | **v1.0** |
| 标签 | `v1.0-lsa-dspine` |
| 创建时间 | 2026-11-18T11:24:16+08:00 |
| 状态 | 已跑通并验证 |
| 快照位置 | 本机 `releases/v1.0-lsa-dspine/`；远端 `/mnt/sda/<user>/releases/v1.0-lsa-dspine/` |
| 单件归档 | `releases/v1.0-lsa-dspine.tar.gz`（5.6 MB，sha256 `4a98441c…09a89`，两侧一致） |
| 校验和 | `MANIFEST.sha256`（1665 个文件，sha256，两侧均已全量校验通过） |
| 源码备份 | `source/` 下 **LSA 已装包 + LSA 上游源码树 + DSpine 全量源码**，共 1649 个文件 |

本版本把「LSA 稀疏注意力 + DSpine 投机解码同时运行」这件事从**未完成**推到**跑通并量化**。
之前卡在 verify 前向最后一步的 `RuntimeError: shape '[2, 24, 256]' is invalid for input of size 98304`。

## 1. 核心修复

**根因**（不是行数公式）：DSpine 每个 verify 步把 16 个 draft token 的 KV 全部写进 KV 池，
但采样只接受其中 `accept_len` 个，被拒的尾巴会被下一个块重写。旧代码让
`index.extend()` 把整块 16 个 token 追加进索引并把该请求的 `end` 指针推进到块尾，
于是下一次 verify 的块起点落在 `end` 之前，`selector.prefill` 里

```python
first = max(end, query_start) - query_start    # 16 行查询里前 14 行被当成"已索引"丢弃
```

只剩 2 行列表，与 16 行 query 对不上。

实测证据（bs=1、块大小 16）：

| 批次 | `seq_lens` | 真实块起点 | 索引 `end` | 选出行数 |
|---|---|---|---|---|
| 第 1 次 verify | 138 | 122 | 122 | **16** ✓ |
| 第 2 次 verify | 140 | 124 | **138** | **2** ✗ |

**修法**：`[PATCH spec v3]` —— verify 批改为**只读选择**，draft 块不进索引：

1. 由本批维护把索引推进到已提交前缀（`ends = seq_lens - k`）；
2. `index.write()` 投影出索引查询后，逐请求 `selector.read_segment()` 只读选择，
   **不调用 `cache.append`**。

**为什么只读等价**（均已读源码确认）：

- 列表读取全程按 **位置 → KV slot** 走 sglang 的 `slot_table`（`req_to_token`），不走 LSA 自己的页表；
- 读取内容 = 分桶 `top_k=4096` ∪ 最近 `window_size=512` 个位置 ∪ `sink_size=4`，
  而 draft 块只有 16 个 token，天然被 dense window 覆盖；
- 被接受的 token **不需要新代码**就进索引：下一次 forward 的 `maintain → cache.advance`
  会按位置把它们的 index key 从 KV 池里读回来。实测 `end` 只随接受量前进：`122 → 124 → 125 → 131`。

## 2. 快照内容（1665 个文件，21 MB）

### 2.1 变更产物与运维文件

| 路径 | 说明 |
|---|---|
| `models/qwen3_5/backend.py` | **已修补的 LSA 后端**（含 `[PATCH spec]` / `[PATCH spec v2]` / `[PATCH spec v3]` / `[LSA-DBG]`） |
| `index/registry.py` | 已修补的索引注册表（`ends_of` 为 verify 合成 `lengths`） |
| `models/qwen3_5/backend.py.upstream`、`index/registry.py.upstream` | 未修补的上游版本，便于 diff |
| `launch/dspine_lsa_v3.sh` | **V1.0 启动脚本**（目标配置：triton + `--page-size 1` + `--disable-cuda-graph` + lsa block） |
| `launch/bf16_src_uds.sh` | 纯 DSpine 基线 |
| `launch/abl_lsa.sh`、`launch/abl_dspine_only.sh` | A/B 对照用（加 `--decode-log-interval 1` 以打印每步 `accept len`） |
| `patches/patch_lsa_spec_v2.py` | 幂等补丁：诊断 + 每请求 token 数取自 `out_cache_loc` |
| `patches/patch_lsa_spec_v3.py` | 幂等补丁：**核心修复**（verify 只读选择） |
| `tools/restart_any.sh` | 重启桥接 + 服务（`[i]pc_bridge` 防自杀写法） |
| `tools/accept_stats.sh` | 从日志统计 accept len / rate / 分布 |
| `restore.sh` | 把快照装回 site-packages（部署或回滚） |
| `VERSION` | 版本元数据 |

### 2.2 完整源码备份（`source/`，1649 个文件）

| 路径 | 原始路径 | 文件数 | 说明 |
|---|---|---|---|
| `source/lsa-installed/` | `site-packages/lsa/` | 348 | **运行时权威代码**：当前进程实际 import 的 LSA 包，含本次全部补丁 |
| `source/lsa-upstream/` | `/mnt/sda/<user>/lsa_code/latent-sparse-attention/` | 1161 | LSA 项目源码树（`src/tests/configs/docs`），干净上游，**不含补丁** |
| `source/dspine/` | `/mnt/sda/<user>/dspine_code/DSpine-main/` | 140 | DSpine 插件源码，以 `pip install -e` 挂进环境 |

配套文件：

| 文件 | 说明 |
|---|---|
| `source/ORIGIN.md` | 三棵树的来源路径、pip 元数据（版本/Location/editable）、未备份内容的说明 |
| `source/patches.diff` | 相对上游的完整 unified diff（181 行，`backend.py` 改 114 行、`registry.py` 改 7 行） |
| `source/MANIFEST.sha256` | 1649 个源文件的校验和 |

**为什么同时留 `lsa-installed` 和 `lsa-upstream`**：本次补丁打的是 site-packages 里的
*已安装包*，上游源码树里没有这些改动；留存干净上游后，`diff -r source/lsa-upstream/src/lsa
source/lsa-installed` 就能直接看出改了什么，`source/patches.diff` 已是该 diff 的固化版本。

**未备份**：`__pycache__`/`*.pyc`（可再生）、模型权重与 KV 缓存、LSA 蒸馏 map、
DSpine 草稿权重 `/mnt/sda/<user>/dspine_draft`（体积大，路径见 `VERSION`）。

### 2.3 归档与校验

```bash
# 单件归档（两边一致）
releases/v1.0-lsa-dspine.tar.gz   5.6 MB   sha256 4a98441cbf1c47df3e9d2f7727f11e786d23bdf22267a5090192b80471aa9a89

# 目录级校验
cd releases/v1.0-lsa-dspine              && sha256sum -c MANIFEST.sha256          # 1665/1665
cd releases/v1.0-lsa-dspine/source       && sha256sum -c MANIFEST.sha256          # 1649/1649
```

备份/打包脚本（可重放）：`tools/add_source_backup.sh`、`tools/make_patch_diff.sh`、
`tools/make_release_v1.sh`。

## 3. 运行方式

```bash
# 远端启动
cd /mnt/sda/<user>
bash restart_any.sh dspine_lsa_v3.sh lsa_final.log

# 本机隧道（每次重启服务后必须重建）
ssh -N -L 127.0.0.1:30000:/mnt/sda/<user>/run/sglang.sock amd03
```

环境：sglang 0.5.22、torch 2.14.1+cu130、lsa 0.1.0、sm120（RTX PRO 6000 Blackwell Max-Q，97,887 MiB）。

## 4. 实测数据（同 prompt、temperature 0、单流）

| 指标 | DSpine-only | LSA+DSpine | 结论 |
|---|---|---|---|
| 投机接受率（服务端 `accept len`） | 5.510 | **5.157** | 93.6% 持平 |
| needle 8k / 32k / 64k | 全命中 | 全命中 | 选择未选错 |
| decode @64k（已扣 prefill） | 90.48 tok/s | 90.89 tok/s | 1.00× |
| **decode @119k** | 63.94 tok/s | **93.22 tok/s** | **1.46×** |
| prefill @119k | 2865 tok/s | **4032 tok/s** | **1.41×** |
| 短上下文五类负载 | 基准 | 0.59–0.88× | 64k 以下 LSA 净亏 |

结论：**稀疏与投机能叠加**，交叉点约 64k；119k 起 decode 反超 46% 且与上下文长度解耦
（64k 90.89 → 119k 93.22）。该结果还是在 LSA 侧扛着 `--disable-cuda-graph` 劣势下取得的。

## 5. 回滚 / 部署

```bash
# 装回 v1.0（或从其他版本回滚到 v1.0）
bash /mnt/sda/<user>/releases/v1.0-lsa-dspine/restore.sh

# 完全回到上游未修补状态
cd /mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang
cp models/qwen3_5/backend.py.orig models/qwen3_5/backend.py
cp index/registry.py.orig index/registry.py

# 恢复到中间态（保留诊断、去掉 v3 只读选择）
cp models/qwen3_5/backend.py.pre-v3 models/qwen3_5/backend.py
```

快照完整性校验：

```bash
cd /mnt/sda/<user>/releases/v1.0-lsa-dspine && sha256sum -c MANIFEST.sha256
```

## 6. 已知限制（下一版要处理）

1. `--disable-cuda-graph` 的代价尚未单独扣除 —— 需要一个「DSpine-only + 同样禁用 CUDA graph + triton」的基线才能干净归因。
2. verify 的 `k = out_cache_loc.numel() // bs` 假设各请求块大小一致（DSpine 恒为 16）；变长块（SPS 策略非均匀）需按请求拆。
3. 并发 bs>1 的 verify、以及前缀复用（`reuse`）在投机下均未专门验证。
4. `read_segment` 的「查询须落在缓存最后推进的页内」这一约束，当前靠 `PAGE_SIZE=8192` + 只走 `slot_table` 规避，改分支需重测。
