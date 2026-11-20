# LSA + 投机草稿（DSpine）联合运行 — 任务交接文档

> 目标：让 LSA（Latent Sparse Attention）与 DSpine 投机解码在 Qwen3.8-27B 上同时运行。
> 状态：**已跑通**（2026-11-18）。verify 批的选择行数与查询行数完全对齐，长上下文 needle 全部命中，
> 投机接受率与无 LSA 时基本持平，119k 上下文下 decode 反超纯投机 1.46×。
>
> **版本基线：`v1.0-lsa-dspine`** — 发布说明见 `reports/release-v1.0-lsa-dspine.md`，
> 带校验和的快照在 `releases/v1.0-lsa-dspine/`（远端 `/mnt/sda/<user>/releases/v1.0-lsa-dspine/`），
> 含 **DSpine 与 LSA 的完整源码备份**（`source/lsa-installed/`、`source/lsa-upstream/`、`source/dspine/`，
> 1649 个文件）以及相对上游的 `source/patches.diff`。
> 回滚/部署：`bash /mnt/sda/<user>/releases/v1.0-lsa-dspine/restore.sh`。

## 0. 环境

| 项 | 值 |
|---|---|
| 远端主机 | `ssh amd03` → `<gpu-host>`（用户 `<user>`） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q，**sm120**，97,887 MiB |
| SGLang python | `/mnt/nvme1n1/sglang522/bin/python`（SGLang 0.5.22, torch 2.14.1+cu130） |
| LSA 包 | `lsa 0.1.0`，装在 `/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/` |
| LSA 蒸馏 map | `/mnt/sda/<user>/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt` |
| DSpine 源码 | `/mnt/sda/<user>/dspine_code/DSpine-main`（已 `pip install -e`） |
| 模型 | BF16 `/mnt/sda/models/Qwen3.8-27B`、`/mnt/sda/<user>/dspine_train/model_view`；NVFP4 `/mnt/sda/models/Qwen3.8-27B-NVFP4` |

### 关键环境限制（重要）

**远端 `<gpu-host>` 的出向 TCP 有精确 100 KiB（102,400 字节）单连接上限**：超过即 `EPERM`。
- 与分块大小、限速、会话无关；走网卡 IP 同样失败
- **AF_UNIX（Unix 域套接字）完全不受影响**（1 MB / 4 MB 均成功）
- 外部主机直连远端 TCP **不受限**（Windows 侧 curl 报错是本地沙箱拦 curl 的假象）

因此远端 SGLang 以 **UDS 监听**，用 `ipc_bridge.py` 把它暴露成 TCP 端口，本机用 SSH StreamLocal 转发。

## 1. 运行方式

### 远端启动（顺序不能反）

```bash
cd /mnt/sda/<user>
bash restart_any.sh dspine_lsa_v3.sh lsa_final.log     # 通用：<启动脚本> <日志名>
# 等价于手写：
#   pkill -9 -f '[i]pc_bridge'; pkill -9 -f '[s]glang_uds'; sleep 6; rm -f run/sglang.sock
#   起 ipc_bridge.py（必须先起：SGLang 启动时有一次内部 warmup 要连 host:port）
#   SGLANG_UDS_PATH=... setsid nohup bash <启动脚本> > logs/<名字>.log 2>&1 &
```

> **踩坑**：`pkill -9 -f ipc_bridge` 会把「正在执行这条命令的 shell 自己」也匹配掉（命令行里含该字符串），
> 表现为 ssh 直接 exit 1 且无输出。必须写成 `pkill -9 -f '[i]pc_bridge'`，或放进脚本文件里执行。

| 脚本 | 内容 |
|---|---|
| **`dspine_lsa_v3.sh`** | **DSpine + LSA（本次目标，已跑通）**：triton + `--page-size 1` + `--disable-prefill-cuda-graph` + `--disable-cuda-graph` + lsa block |
| `bf16_src_uds.sh` | BF16 + 源 DSpine（纯投机基线） |
| `lsa_uds.sh` | BF16 + LSA 单独（131k 上下文） |
| `abl_dspine_only.sh` / `abl_lsa.sh` | 两个 A/B 对照用：在对应脚本上加 `--decode-log-interval 1`，让日志每步打印 `accept len` |
| `nextn_uds.sh` / `nextn16_uds.sh` / `verify_ngram16b10.sh` | NEXTN(MTP)、NGRAM 等其他投机 |

### 本机（Windows）隧道

```powershell
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
    -L 127.0.0.1:30000:/mnt/sda/<user>/run/sglang.sock amd03
```

**每次远端重启服务后必须重建隧道**（服务重启会删除并重建 UDS，旧隧道指向失效 inode）。

## 2. 已打的补丁（全部在 LSA 包内，均有 `.orig` / `.pre-v3` 备份）

文件：
- `.../lsa/sglang/models/qwen3_5/backend.py`（`[PATCH spec]` / `[PATCH spec v2]` / `[PATCH spec v3]` / `[LSA-DBG]`）
- `.../lsa/sglang/index/registry.py`（`[PATCH spec]`）

用 `grep -n "\[PATCH spec\|\[LSA-DBG" <文件>` 可一次列出。

| # | 位置 | 改动 |
|---|---|---|
| 1 | `backend.py` `refuse_unservable` | 删除 `speculative_algorithm is not None → raise NotImplementedError` |
| 2 | `backend.py` `select()` | extend / verify 统一走索引推进（后续被 v3 拆开） |
| 3 | `backend.py` `select()` | verify 批缺 `lsa_rows` 时就地 `registry.maintain(forward_batch)` |
| 4 | `backend.py` `select()` | verify 合成 `lengths`/`prefix`（v2 改为从 `out_cache_loc` 推） |
| 5 | `backend.py` `select_and_read()` | 删除 verify / draft-extend 的 `NotImplementedError` |
| 6 | `registry.py` `ends_of()` | verify 合成 `lengths`（`positions.numel() // request_count`） |
| 7 | `backend.py` `select()` | `[LSA-DBG]` 一次性诊断（`LSA_SPEC_DEBUG=1`） |
| 8 | `backend.py` | **v2**：`dbg()` / `cache_ends()` 诊断助手；verify 的每请求 token 数改由 `out_cache_loc` 的新 token 数推算 |
| 9 | `backend.py` `select()` | **v3**：**verify 改为只读选择，draft 块的 KV 不再写进索引**（核心修复） |

补丁脚本（可重放、幂等）：`/mnt/sda/<user>/lsa_code/patch_lsa_spec_v2.py`、`patch_lsa_spec_v3.py`。

### 回滚

```bash
cd /mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang
cp models/qwen3_5/backend.py.orig models/qwen3_5/backend.py   # 回到完全原始
# 或只回到 v2（保留诊断、去掉 v3 只读选择）：
cp models/qwen3_5/backend.py.pre-v3 models/qwen3_5/backend.py
cp index/registry.py.orig index/registry.py
```

## 3. 根因（已定位并修复）

### 3.1 现象

```
backend.py:536 triton_sparse_read
q.contiguous().view(token_count, layer.tp_q_head_num, layer.head_dim)
RuntimeError: shape '[2, 24, 256]' is invalid for input of size 98304
```
`token_count = lists.shape[0] = 2`，而 `q` 实际有 `98304 / 6144 = 16` 行。

### 3.2 真正的原因（不是「行数公式」问题）

`index.extend()` **本来就是按 `lengths` 逐请求切片查询行**的（`selector.prefill` 按
`query_start/query_count` 切，`read_segment` 每行查询返回一行列表），所以只要传对 `lengths`
就会输出 `sum(lengths)` 行。真正砍掉行的是 **索引自身的 `end` 指针超前**。

v2 诊断补丁打出的实测数据（`LSA_SPEC_DEBUG=1`，bs=1、块大小 16）：

| 批次 | `seq_lens` | 真实块起点 `seq_lens-16` | `positions[0]` | 索引 `end` | 选出列表行数 |
|---|---|---|---|---|---|
| 第 1 次 verify | 138 | 122 | 122 | **122** | **16** ✓ |
| 第 2 次 verify | 140 | 124 | 124 | **138** | **2** ✗ |

`selector.prefill` 里

```python
first = max(end, query_start) - query_start      # end=138, query_start=124 -> first=14
stop  = min(end + take, query_start + query_count) - query_start   # stop=16
```

于是 16 行查询里前 14 行被当成「已经索引过的」直接丢弃，只剩 2 行 → `cat` 出 2 行。

**为什么 `end` 会超前**：DSpine 每个 verify 步把 16 个 draft token 的 KV 全部写进池子，
但采样只接受其中 `accept_len` 个（上例 accept=2），被拒的尾巴会被下一个块重写。
旧写法让 `extend` 把整块 16 个 token 追加进索引并推进 `end`，下一次块的起点（124）
就落在 `end`（138）之前 —— 索引「以为」自己已经见过这些位置了。

### 3.3 修复：verify 是只读选择，不进索引

`[PATCH spec v3]` 之后 verify 分支只做两件事：

1. 由本批的维护把索引推进到**已提交前缀**（`ends = seq_lens - k`，即 `selector.prefill` 的起点）；
2. **只读**地按位置做选择，不调用 `cache.append`：

```python
index_queries = index.write(inputs, slots)
for _row, _count in zip(forward_batch.lsa_rows, [_k] * _bs, strict=True):
    lists.append(index.selector.read_segment(_row, index_queries[a:b], positions[a:b]))
```

**为什么只读是对的**（关键依据，都已读源码确认）：

- 列表的读取全部是**位置 → KV slot** 走 sglang 的 `slot_table`（`req_to_token`），
  不走 LSA 自己的页表：`expand_by_threshold` 里 `MAP_SLOTS` 分支把每个位置读成 slot。
- 读取内容 = 分桶选出的 `top_k` ∪ **最近 `window_size` 个位置** ∪ `sink_size`。
  本设计实测参数：`top_k=4096`、`window_size=512`、`sink_size=4`、`bucket_size=8`、
  `cluster_block_size=512`、`PAGE_SIZE=8192`、`top_level=0`（无层级树）。
  `width = 4096 + 512 + 4 = 4612`。
- 因此 draft 块内部的前序 token 由 512 长度的 dense window 覆盖（块只有 16），
  不需要把块写进索引；每行最多漏掉与其块内偏移相等的 ≤15 个位置，可忽略。
- **被接受的 token 不需要新代码**就会进索引：下一次 forward 的 `maintain` →
  `cache.advance(end)` 会按位置把它们的 index key **从 KV 池里读回来**建索引。
  这正好是用户最初提出的两步方案，且实测 `end` 只随接受量前进：
  `122 → 124 → 125 → 131`（对应 `seq_lens` 138 → 140 → 141 → 147）。

## 4. LSA 索引构建方式与维度（已完整读通）

| 量 | 值 |
|---|---|
| `index_dim` | **64**（`W_Q_idx`/`W_K_idx` 都是 `(768, 64)`） |
| 输入投影维 | 768 = 4 kv head × 192 position-free dims（Q 侧 = 每组 6 个 query 求和） |
| `kv_head_count` / `head_dim` | 24 / 256 |
| 列表宽度 `width` | **4612** = `top_k 4096 + window 512 + sink 4` |
| 索引键 | `[M, 64]` bf16，单位行 |
| 桶质心 | `[rows, 64]` bf16 |

链路：`index_space.key_and_query()` → 每层 `W_K_idx`/`W_Q_idx`(768→64) →
`LayerIndex.extend(inputs, positions, slots, rows, lengths, starts)` →
`selector.prefill`（逐请求、按页增长地 append + read）→ `[S, width]` int32 KV slot 列表 →
`triton_sparse_read` 做 paged 稀疏注意力。

## 5. 已确认的事实（避免重复踩坑）

1. **`ForwardMode.is_extend()` 本来就包含 `TARGET_VERIFY`**（`forward_batch_info.py:230`）。
2. **verify 批不携带 `extend_seq_lens_cpu` / `extend_prefix_lens_cpu`**，需用 `seq_lens - k` 合成；
   `k` 应取自 `forward_batch.out_cache_loc.numel() // bs`（新 token 数），不要用 `positions.numel()`。
3. **LSA 的选择在 host 侧做，不能被 CUDA graph 捕获** → 必须 `--disable-cuda-graph`（verify 也算 extend）。
4. **DSpine 不往目标模型 KV 写任何东西**（草稿有自己的池），所以「索引失配」不存在。
5. LSA 只钩 `FlashAttentionBackend` / `TritonAttnBackend`；sm120 上 FA3 不可用，**只能用 triton**。
6. LSA 对 Qwen 拒绝 context parallel（`attn_cp_size > 1` 或 `enable_prefill_cp`）。
7. 启动链需要 `if __name__ == "__main__":` 守卫（`sglang_uds.py`、任何 in-process Engine 脚本同理）。
8. 包装层 `hybrid_linear_attn_backend.forward_extend` **不改写 `forward_batch`**，`positions` 原样透传。
9. `pkill -f <含自身命令行的字符串>` 会杀掉执行脚本的 shell，必须用 `[i]pc_bridge` 写法。

## 6. 量化结果（2026-11-18 实测，同 prompt、temperature 0、单流）

### 6.1 短上下文五类负载（`max_tokens=400`，`tools/bench5.py`）

| 负载 | DSpine-only tok/s | LSA+DSpine tok/s | 比 |
|---|---|---|---|
| A 自由问答 | 52.60 | 42.16 | 0.80× |
| B 逐字引用 | 179.72 | 153.32 | 0.85× |
| C 代码 | 118.03 | 69.89 | 0.59× |
| D JSON | 161.77 | 141.68 | 0.88× |
| E 推理 | 117.69 | 93.19 | 0.79× |

短上下文 LSA 只会亏：上下文装得进 512 window，稀疏省不下算力，却要付选择开销。
注意 LSA 侧还带着 `--disable-cuda-graph` + triton 后端的额外劣势。

### 6.2 投机接受率（服务端日志 `accept len`，`--decode-log-interval 1`，`tools/accept_stats.sh`）

| 配置 | steps | mean accept | rate | median | p90 | 满接受步数 |
|---|---|---|---|---|---|---|
| DSpine-only | 337 | **5.510** | 0.301 | 3.0 | 16.0 | 42 |
| LSA+DSpine | 362 | **5.157** | 0.277 | 3.0 | 15.0 | 36 |

**接受率 93.6% 持平** → v3 的 verify 选择在语义上是等价的，没有选错 KV。
（长上下文 needle 系列：LSA 4.772 vs DSpine 4.892，97.5%。）

### 6.3 needle 正确性（`tools/needle_test.py`，针埋在上下文正中）

| 上下文 | LSA+DSpine | DSpine-only |
|---|---|---|
| 8k | 命中 ✓ | 命中 ✓ |
| 32k | 命中 ✓ | 命中 ✓ |
| 64k | 命中 ✓ | 命中 ✓ |

### 6.4 长上下文 decode（先用同 prompt 预热前缀缓存，把 prefill 从墙钟里去掉）

| 上下文 | LSA+DSpine | DSpine-only | 比 |
|---|---|---|---|
| 64k | 90.89 tok/s | 90.48 tok/s | 1.00× |
| **119k** | **93.22 tok/s** | **63.94 tok/s** | **1.46×** |

prefill @119k：LSA **29.51 s（4032 tok/s）** vs DSpine 41.53 s（2865 tok/s）= **1.41×**。

**结论：稀疏与投机确实能叠加。** 交叉点在 64k 附近；64k 以下 LSA 净亏，
119k 起 LSA 开始回本并在 decode 上反超 46%，且其 decode 速率基本与上下文长度无关
（64k 90.89 → 119k 93.22）。

## 7. 建议的验收顺序（已全部完成）

1. ✅ `dspine_lsa_v3.sh` 启动：`lsa: enabled` + `DSPINE ready` + `ready to roll`
2. ✅ 短上下文 5 类负载：accept 5.157 vs 5.510（93.6%）
3. ✅ 8k / 32k / 64k needle 全部命中
4. ✅ 64k / 119k：prefill 4032 vs 2865、decode 93.22 vs 63.94
5. ✅ 三方对比见 §6

## 8. 未决风险

- **`--disable-cuda-graph` 的代价还没单独扣掉**。下一个实验应该是「DSpine-only + 同样禁用
  CUDA graph + triton 后端」的基线，才能把 §6.1 / §6.4 的差距干净地归因给 LSA。
- verify 的 `k = out_cache_loc.numel() // bs` 假设各请求块大小一致（对 DSpine 恒为 16，成立）。
  若将来出现变长块（SPS 策略 `verify_widths` 非均匀），这里要按请求拆。
- verify 只读路径**不调用 `cache.settle`**。因为不 append、不关 cluster block，当前是自洽的；
  但如果将来改成「只在块内位置已跨过 512 边界时才 settle」，要重新验证。
- `read_segment` 的 docstring 说查询应落在「缓存最后推进到的那个页里」。当前 `PAGE_SIZE=8192`、
  块只有 16、且读取全走 `slot_table`，所以跨页无影响；若改成 `block_positions` 分支需重测。
- 前缀复用（`reuse`）路径未在投机下专门验证：verify 不推进 `closed_keys`，指纹只对已提交前缀生成，
  理论上更安全，但值得用两条相同长前缀的并发请求验一次。
- 并发场景（bs>1 verify）只在小 bs 下验证过；`rows`/`lengths` 逐请求切片逻辑是按 bs 循环写的，
  但没跑 `max-running-requests` 满载。

## 9. 本次新增文件

远端 `/mnt/sda/<user>/`：
- `restart_any.sh <启动脚本> <日志名>` — 重启桥接 + 服务（含 `[i]pc_bridge` 防自杀写法）
- `accept_stats.sh <日志> [起始行]` — 从日志统计 accept len / rate / 分布
- `abl_dspine_only.sh`、`abl_lsa.sh` — A/B 对照启动脚本（带 `--decode-log-interval 1`）
- `lsa_code/patch_lsa_spec_v2.py`、`lsa_code/patch_lsa_spec_v3.py` — 幂等补丁脚本
- `make_release_v1.sh` — 打包 v1.0 快照（变更产物 + 运维脚本 + 校验和）
- `add_source_backup.sh` — 把 LSA 已装包 / LSA 上游树 / DSpine 源码全量备份进 `source/`
- `make_patch_diff.sh` — 生成 `source/patches.diff` 并重算清单、打 tarball

本机 `tools/`：
- `bench5.py` — 五类负载单流基准（结果写 `bench5.json`）
- `needle_test.py` — 长上下文 needle 正确性
- `decode_probe.py` — 用前缀缓存扣掉 prefill，单测长上下文 decode 速率
- `accept_stats.sh`、`restart_any.sh` — 与远端同名脚本的源文件
- `make_release_v1.sh`、`add_source_backup.sh`、`make_patch_diff.sh` — 版本打包三件套

## 10. 版本归档结构（v1.0-lsa-dspine）


```
releases/v1.0-lsa-dspine/               1665 files  21 MB  (本机与远端逐字节一致)
├── VERSION                             版本元数据
├── MANIFEST.sha256                     1665 项校验和（两侧均全量通过）
├── restore.sh                          装回 site-packages
├── models/qwen3_5/backend.py           已修补后端 + .upstream 对照
├── index/registry.py                   已修补注册表 + .upstream 对照
├── launch/{dspine_lsa_v3,bf16_src_uds,abl_lsa,abl_dspine_only}.sh
├── patches/{patch_lsa_spec_v2,patch_lsa_spec_v3}.py
├── tools/{restart_any,accept_stats}.sh
└── source/                             1649 files —— 完整源码备份
    ├── lsa-installed/                  348 files  运行时权威代码（含补丁）
    ├── lsa-upstream/                  1161 files  干净上游源码树
    ├── dspine/                         140 files  DSpine 插件源码
    ├── ORIGIN.md                       来源 / pip 元数据 / 未备份说明
    ├── patches.diff                    相对上游的 unified diff（181 行）
    └── MANIFEST.sha256                 1649 项校验和

releases/v1.0-lsa-dspine.tar.gz          5.6 MB   sha256 4a98441c…09a89
```

---

## 11. 优化 #1：拿回 target-verify CUDA graph（已完成）

目标：让 verify 批跑进 CUDA graph。收益实测 **@119k decode +15.6%**。

### 11.1 三个阻塞点（逐个由日志证实，不是推测）

| # | 现象 | 根因 |
|---|---|---|
| 1 | `RuntimeError: LSA maintenance needs request ids on the forward batch` | 我们自己加的 `maintain` 兜底在**捕获前的预热 pass** 触发；该 pass 没有 `rids`，也不在 `is_current_stream_capturing()` 状态，所以捕获标志挡不住 |
| 2 | `shape '[13, 24, 256]' is invalid for input of size 1179648` | `Selector.__init__` 的 `decode_top_k = TopK(top_p, request_count, ...)`：`TopK.listed` 用 `held[:rows]` 从**按请求表行数（=13）定尺**的缓冲切片，喂 192 行被静默截成 13 行 |
| 3 | `LSA selects a prefill on the host` | `select_and_read` 的捕获拦截用 `mode.is_extend()`，而它包含 `TARGET_VERIFY` |

### 11.2 改法

**v4**（`backend.py`）：verify 分支从「逐请求 host 循环 `read_segment`」改为共用 decode 的设备侧读取：

```python
_rows = forward_batch.req_pool_indices.long()      # 设备张量，非 host list
_k = int(positions.numel()) // int(_rows.numel())
index_queries = index.write(inputs, slots)
return index.selector.decode(index_queries, positions, _rows.repeat_interleave(_k))
```

一个请求的 k 行共用同一张请求表行，所以都读该请求的**已提交前缀**；块内前序 token 由读取自带的
`window_size=512` dense 尾覆盖。该路径（`decode_flat`）本来就是 LSA 单独服务时跑在捕获图里的那条，
因此天然可捕获：`decode_counts` 读 `closed_keys_device`，`scan_paged` 是定形 kernel。

**v5**（`backend.py`）：删掉 verify 分支的 `maintain` 兜底。LSA 原本就只在两个元数据 hook 里维护
（`init_forward_metadata` 与 `init_forward_metadata_out_graph`，后者捕获期跳过、回放前执行），
v4 的读法已不需要 `lsa_rows`。extend 分支的兜底加了 `rids` 判空。

**v7**（`selector.py`）：decode listing 缓冲按「一请求一整个 draft 块」定尺：

```python
self.decode_top_k = TopK(top_p, request_count * draft_rows(), device, exact=True)
```

`draft_rows()` 从 `get_spec()` 取 `speculative_num_draft_tokens`（无投机时为 1）。

**启动**：去掉 `--disable-cuda-graph`，保留 `--disable-prefill-cuda-graph`（prefill 仍在 host 侧选择）。

### 11.3 结果（同代码、同机、多次重复）

@119k decode（4–5 次取中位数）：

| 配置 | 各次 tok/s | 中位数 |
|---|---|---|
| 图关闭 | 82.78 / 81.26 / 80.12 / 81.43 | **81.35** |
| **图开启** | 93.01 / 97.79 / 97.56 / 87.69 / 94.55 | **94.55** |

两组不重叠，**+15.6%**。

短上下文五类负载（同条件背靠背、无每步日志）：

| 负载 | 无图 | 有图 | 提升 |
|---|---|---|---|
| A 自由问答 | 50.93 | 51.66 | +1.4% |
| B 逐字引用 | 178.84 | 186.80 | +4.5% |
| C 代码 | 81.65 | 86.86 | +6.4% |
| D JSON | 162.10 | 162.52 | +0.3% |
| E 推理 | 107.59 | 110.77 | +3.0% |

短上下文只 +3.1%：那里算力大头不在 LSA 选择，而在模型本身。图的收益与 CPU 压力正相关——
带 `--decode-log-interval 1`（CPU 更忙）时同一组 A/B 是 +10~17%，因为省掉的是每步上百次 launch。

### 11.4 正确性

- accept len（`--decode-log-interval 1`）：无图 **5.157**，有图 **5.024**；无图的 5.157 与 v1.0（v3 读法）
  逐位相同 → 换成 decode 读法**质量中立**；有图差 2.6%，在批次噪声内。
- needle 8k / 32k / 64k：有图时**全部命中** → 图回放读到的是新鲜索引状态（out-graph hook 生效）。

### 11.5 遗留与下一步

- **疑点**：v1.0（v3 的 `read_segment` 读法、无图）曾单次测得 @119k **93.22**，而现在 v4 读法无图是 81.35。
  若该单次值可信，则 v4 的 `decode_flat` 比 v3 的 `read_flat` 慢 ~13%，原因是：
  `read_flat` 在桶数越过 `SCAN_SELECT_BUCKETS` 时用**融合**的 `scan_select.listed`（扫描即出列表，不落地 [B, max_rows] 分数矩阵），
  而 `decode_flat` 走 `scan_paged` + 独立 `decode_top_k`，会物化 **[192, 16384] fp32 ≈ 12.6 MB** 的分数矩阵。
  这条要复核，并直接指向 #2 的正确做法。
- **#2 的方向已明确**：verify 读改为「按请求（bs 行）做融合 `scan_select` 一次 → listing 广播到 k 行 → 逐行 expansion」。
  既省 k 倍的桶扫描，又拿回融合 kernel、去掉分数矩阵；且仍是设备侧、可捕获。

### 11.6 新增启动脚本

| 脚本 | 内容 |
|---|---|
| `graph_lsa.sh` | **当前最优**：v1.0 配置去掉 `--disable-cuda-graph`（verify 图开启） |
| `nograph_lsa.sh` | v1.0 原配置（无图），A/B 基线 |
| `graph_lsa_log.sh` / `nograph_lsa_log.sh` | 上两者加 `--decode-log-interval 1`，用于测 accept |

---

## 12. 优化 #2 / #3 / #4 的实测否决（含成本模型修正）

这三项的前提都被数据否掉了，记录在此以免重复投入。

### 12.1 成本模型修正：步耗时是**平坦**的

每步耗时（已用同一 run 的 accept 扣掉接受率差异，图开启、每步日志）：

| 上下文 | 桶数（可见） | 每步耗时 |
|---|---|---|
| 16k | ~2000 | 56.8 / 58.3 ms |
| 64k | ~7900 | 59.8 / 54.5 ms |
| 120k | ~14870 | 58.8 / 55.1 ms |
| 168 token（短） | 0 | ~27 ms |

即：一旦上下文越过 dense window（512），步耗时**跳 ~29ms 后完全饱和**，之后与上下文长度无关。
（这也解释了为什么早期用 tok/s 看会出现"16k 比 120k 还慢"的假象——那纯粹是 accept 差异：3.74 vs 4.94。）

### 12.2 三个否证实验

| 实验 | 变量 | 结果 | 判定 |
|---|---|---|---|
| 桶扫描量 | 2000 桶 vs 14870 桶（7×） | 步耗时相同（见上表） | **扫描不是瓶颈 → #2 前提不成立** |
| 列表宽度 | `top_k` 4096→1024（width 4612→1540，3×） | 步 60.5 / 66.4 ms vs 基线 55.1 / 58.8 | **无改善 → #4 对 tok/s 无收益** |
| 融合 select | `kernels="cutedsl"`（扫完即出列表） | 步 60.9 / 64.2 / 63.5 ms | **慢 ~9% → #3 的 cutedsl 是回退** |

> cutedsl 那组 tok/s 看起来是 +5%（100.8/99.8/99.5 vs 93~97），但它的 accept 是 6.13/6.41/6.32
> 而基线是 4.40/4.94——差异来自内容而非内核，故按**步耗时**判定为回退。

探针用法（可复现，不需改代码）：

```bash
/mnt/nvme1n1/sglang522/bin/python make_lsa_probe.py graph_lsa_log.sh probe.sh top_k=1024
/mnt/nvme1n1/sglang522/bin/python make_lsa_probe.py graph_lsa_log.sh probe.sh kernels=cutedsl
# lsa block 是设计的扁平体：indexer 侧 knobs（top_k/window_size/top_p/sink_size/...）
# 与 serving 侧 knobs（projections/kernels/decode/prefix_reuse/...）同级。
# 放错层级会被 parse 拒绝并列出全部合法键名。
```

### 12.3 那 29ms 最可能是什么

- **DSpine 草稿模型的 dense attention**（`speculative_draft_window_size=None`，即全上下文）。
  LSA 只索引目标模型，草稿自己有 KV 池；给草稿加上下文窗口需要改 DSpine——
  `--speculative-draft-window-size` 已被插件明确拒绝：`ValueError: DSPINE does not support --speculative-draft-window-size.`
- **权重流**：27B BF16 ≈ 54GB，每步至少全量读一遍。短上下文步 27ms ≈ 2 TB/s，已经贴着该卡带宽上限，
  说明短上下文侧没有多少空间；长上下文的额外 29ms 只能靠上面两条之一。
- 要定位必须用 profiler（nsys / ncu）。**不要再用黑盒 CUDA-event 探针**：v8 诊断在 eager 下
  报出 read=40ms/层（×16 层 ≫ 整步 56ms），与 12.2 的两个否证互相矛盾，已放弃并回滚。

### 12.4 运营注意：可用上下文长度随启动波动

`--context-length 131072` 只是上限；**实际可接受输入长度取决于本 boot 分到的 KV 预算**，
实测见过 119005 可用，也见过上限只有 116552（请求被 400 拒：
`Input length (119005 tokens) exceeds the maximum allowed length (116552 tokens)`）。
长上下文基准要么每次都读实际上限，要么留足余量（110k 仍可能与 radix 抖动叠加导致数据不可用）。

### 12.5 当前最优配置

`graph_lsa.sh` = v1.0 配置 + `--disable-cuda-graph` 换成保留 verify 图（triton kernels，`top_k` 默认 4096）：

| 场景 | 结果 |
|---|---|
| 短上下文 B 负载（逐字引用） | **186.8 ~ 189.8 tok/s** |
| @119k decode（n=5 中位） | **94.55 tok/s**（无图 81.35，+15.6%） |
| accept len | 5.02 ~ 5.16 |
| needle 8k / 32k / 64k | 全部命中 |

---

## 13. 决定性归因：那 29ms 不在 LSA，在 DSpine 侧

用「关掉投机、只跑 LSA」做对照（`lsa_uds.sh`，1 token/步，显存/内核与前者同族，decode 图开启）：

| 上下文 | LSA 单独 tok/s | 每步耗时 | 相对短上下文 |
|---|---|---|---|
| 168 token | 25.15 | 39.8 ms | — |
| 16k | 24.24 / 23.92 | 41.3 / 41.8 ms | +5% |
| 100k | 23.10 / 23.09 | 43.3 ms | **+8.8%** |

而 **LSA + DSpine** 是 27 ms（短）→ 56 ms（≥16k），**+107%**。

**结论：LSA 的每步成本是上下文平坦的（短→100k 只涨 8.8%）；LSA+DSpine 那 29 ms 的跳变全部来自
DSpine 侧**——草稿模型的 dense attention（`speculative_draft_window_size=None` = 全上下文）
加上 verify 的 ReplaySSM。这与 §12.2 的两个否证互相印证（改桶数、改列表宽度都不动步耗时）。

由此 **#6（把 LSA 读取移植到 flashinfer paged decode）也应当否决**：读取既不是瓶颈，
移植只能改变实现族而不改变 tok/s，而成本是跨模块移植 + 每步 CSR 转换。

### 13.1 剩余 tok/s 空间的三个方向（都在 LSA 之外）

| 方向 | 预期 | 前置条件 |
|---|---|---|
| 给 DSpine 草稿加上下文窗口/稀疏 | 最大（那 29 ms 的大头） | 必须改 DSpine 源码——`--speculative-draft-window-size` 被插件硬拒绝（`spec.py:71`），且其 verify 依赖 dense draft block 的一致性校验 |
| 权重流（NVFP4） | 短上下文已贴 ~1.4–2 TB/s 带宽，量化可望大幅改善 | `/mnt/sda/models/Qwen3.8-27B-NVFP4` 已在；但 DSpine 明确拒绝 FP4 KV cache，需单独验证 LSA 索引几何与精度 |
| 提高 accept（DSpine SPS/draft token 数） | 线性提升 tok/s | DSpine 侧策略调参 |

---

## 14. FP4（量化权重）实测：责任不在 LSA，FP4 基线自身就是坏的

问题背景：能否把目标模型换成 `/mnt/sda/models/Qwen3.8-27B-NVFP4` 拿到额外 tok/s。

### 14.1 这个 checkpoint 到底是什么

| 项 | 值 |
|---|---|
| 架构 | 64 层、24 q heads / 4 kv heads、head_dim 256、hidden 5120（与 LSA 几何一致） |
| 量化 | `self_attn.(q|k|v|o)_proj`、`linear_attn.*`、`lm_head`、56–63 层 MLP = **FP8(W8A8)**；其余 MLP/MoE = **NVFP4(W4A4, group 16)** |
| MTP 权重 | **存在**（`mtp.fc`、`mtp.layers.0.*` 共 15 个张量），且被 `re:^mtp.*` 排除在量化之外 |
| 体积 | 21.8 GB（bf16 约 54 GB） |

**注意：LSA 读到的 q/k 来自 FP8 投影，不是 4-bit**——量化误差远小于"全 4-bit"的直觉。

### 14.2 LSA 与 FP4 的接口：有一个硬门，可以绕过

`--kv-cache-dtype auto` 在这个 FP8 checkpoint 上把 KV cache 解析成 `torch.float8_e4m3fn`，
而 LSA 的 `key_buffer()` 硬性要求 bf16（索引 map 在 bf16 键上蒸馏）：

```
NotImplementedError: the key and query input reads a bf16 KV cache -
this one is torch.float8_e4m3fn, serve kv_cache_dtype auto
```

显式 `--kv-cache-dtype bfloat16` 即可放行：LSA 能加载、能捕获 target-verify 图、能正常推理。
**所以 LSA 本身不排斥 FP4 权重。**

### 14.3 之前 FP4 是怎么部署的（远端既有脚本，11-16～11-17）

远端早就有 FP4 的启动脚本，都不是本次会话建的：

| 脚本 | 时间 | 投机 | 对应日志 |
|---|---|---|---|
| `verify_ar.sh` | 11-17 15:43 | **无投机（AR）** | `ar.log` ready=1 errors=0 |
| `verify_ngram.sh` | 11-17 15:40 | NGRAM(8) | — |
| `verify_dspine.sh` | 11-17 11:32 | DSpine(`dspine_draft`) | — |
| `verify_dspine_mine.sh` | 11-17 11:47 | DSpine(`dspine_final_v3`) | — |
| `run_dspine.sh` | 11-16 13:43 | DSpine | — |

它们共同的参数特征（**这就是"之前能跑"的配方**）：

```
--attention-backend flashinfer
--linear-attn-backend triton
--mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 64
--mem-fraction-static 0.84        (run_dspine.sh 是 0.78)
--context-length 131072
--cuda-graph-max-bs-decode 32
--min-free-slots-delay 1 --sampling-backend flashinfer --trust-remote-code
--host 0.0.0.0 --port 30000
```

**关键：没有 `--kv-cache-dtype`、没有 `--page-size`、没有 `--quantization`** —— 即全部走默认，
KV cache 由 `auto` 解析成 **fp8**。

### 14.4 修正：FP4 本身没问题，是"LSA 强制参数"毒化了它

按上面的原始参数复跑（只把 `sglang serve` 换成 UDS 启动器，其余一字不改）：
**输出完全正常**——逐字引用负载原文复现（`...Memphis and New Orleans.`），needle 8k 命中 `BLUEHERON-7742`。

再做逐项隔离（以"原部署参数"为基线，每次只加一个 LSA 所需参数）：

| FP4 配置 | 输出 |
|---|---|
| 原部署参数（默认 KV=fp8） | ✅ 正常 |
| **+ `--page-size 1`** | ✅ 正常（原文复现、needle 命中） |
| **+ `--kv-cache-dtype bfloat16`** | ❌ 崩坏（400 token 空内容、needle 未命中、乱码） |
| + LSA 全套（bf16 KV + page-size 1 + triton） | ❌ 崩坏（与前一行同因） |

**结论：`--kv-cache-dtype bfloat16` 是唯一的毒点，而这个 flag 正是 LSA 硬性要求的**：

- LSA 的 `key_buffer()` 要求 KV cache buffer 是 bf16（`NotImplementedError: ... this one is torch.float8_e4m3fn`），
  因为索引 map 是在 **bf16 键**上蒸馏的；
- 这个 checkpoint 是 W8A8（`q/k/v/o_proj` = FP8 + 逐 token 激活量化）的混合精度包，
  它的注意力路径把 KV 当 fp8 用；强行改成 bf16 会把注意力输出毒化，输出随即崩坏。

⇒ **在当前实现下，LSA 与这个 FP4 checkpoint 是互斥的**，而且不是"收益变小"，是**无法正确运行**。
要共存只有两条路：
1. **在 fp8 KV 键上重新蒸馏索引 map**，并让 LSA 支持读 fp8 KV cache（改动大，需重新训练 M_Wq/M_Wk）；
2. 造一个 **KV cache 保持 bf16 且注意力仍正确**的 FP4 包（即量化方案不能像现在这样把 KV 拉进 fp8 契约）——
   这属于量化侧工作，与本仓库的 LSA 改动无关。

> 本次会话早前"FP4 基线自身就是坏的"的判断是**错的**，已由本节修正：那批崩坏是我自己强加
> `--kv-cache-dtype bfloat16` 导致的。FP4 用它的原生参数是好的。

### 14.5 顺带修掉的一个真实缺口：LSA × NEXTN/MTP（patch v9）

MTP 在 `eagle_prepare_for_verify → decode_cuda_graph_runner.load_batch` 里构造的**回放视图不带 host 侧
`seq_lens_cpu`**，导致 LSA 的维护 hook 崩溃（与 MTP 语义无关，纯粹是数据来源问题）：

```
File ".../lsa/sglang/index/registry.py", line 77, in ends_of
    seq_lens = forward_batch.seq_lens_cpu.tolist()[:request_count]
AttributeError: 'NoneType' object has no attribute 'tolist'
```

`[PATCH spec v9]` 让 `ends_of()` 在缺 host 副本时回落到设备张量 `forward_batch.seq_lens`
（该 hook 在捕获外执行，同步合法）。打过之后 FP4+LSA+MTP 能正常启动并捕获 verify 图。
**该补丁尚未收进 v1.1 快照**（v1.1 的 registry.py 还没有 v9），下次出包时一并纳入。

---

## 15. 四路吞吐分解（dense / +LSA / +DSpine / +LSA+DSpine）与两处结论更正

完整报告：`reports/qwen38-four-way-decomposition.md`。测量规范：无逐字日志、每配置独立重启、
长上下文一律生成 399 token（`decode_probe.py <ctx> 400 long`）。

### 15.1 tok/s

| 配置 | 五类负载均值 | 相对 dense |
|---|---|---|
| dense（flashinfer） | 24.72 | 1.00× |
| +LSA（triton/page1） | 24.56 | 0.99× |
| +DSpine（flashinfer） | 130.58 | **5.28×** |
| +LSA+DSpine | 123.64 | 5.00× |

长上下文（399 token 解码，同一 prompt 内可比）：16k / 64k / 118k 分别为
dense 24.42 / 23.04 / 21.83；+LSA 24.27 / 23.46 / 23.29；
+DSpine 70.9–74.9 / 87.5–88.5 / 72.7；+LSA+DSpine 66.8–69.6 / 79.2–82.1 / 66.4–73.8。

### 15.2 步耗时归因（ms/step，batch=1，`accept/throughput`，与 accept 无关）

| 上下文 | DSpine+flashinfer | triton+page1（无 LSA） | LSA+DSpine |
|---|---|---|---|
| 16k | 未测 | 80.6 | **52.2** |
| 64k | 55.6 | 180.5 | **53.8** |
| 118k | 59.6 | 296.7 | **54.3** |

关键对照（新增 `ctrl_ds_triton_p1.sh`）：**LSA 强制的 triton+page1 后端在无索引时随上下文线性恶化到
296.7 ms/步**，LSA 把它拉回 54.3（5.5× 修复），并比不开 LSA 的最优配置 flashinfer 还快 **8.9%（118k）/ 3.2%（64k）**。
这就是 LSA 在这套系统里的真实定位：不是"稀疏比稠密快"，而是"page_size=1 的 paged gather 代价随上下文爆炸，
稀疏索引把代价钉死在 top_k=4096"。

### 15.3 更正 §12.5

| 原说法 | 更正 |
|---|---|
| `graph_lsa.sh` @119k decode = **94.55 tok/s**（n=5 中位，无图 81.35，+15.6%） | **不可复现**。同一脚本同日 3 次读数 74.67 / 74.67 / 60.22。步耗时仍是 **54.3 ms**（与 12.1 记录的 54 ms 一致），因此差异全部来自 accept，配置性能没有退化 |
| "LSA 在 119k 给 DSpine 带来 1.29–1.46×" | **不成立**。该倍数来自 tok/s 跨 prompt 比较。按步耗时，LSA 的收益是 **8.9%/步（118k）**，accept 不变时约合 tok/s +9% |
| 早期 decode 探针（只生成 50 余 token） | **不要再用于任何结论**：首步开销占分母，读数可在 60–98 tok/s 间摆动 |

> 同一 prompt 下的步耗时才是可比量；tok/s 跨上下文/跨 prompt 比较只反映 accept 差异（已实测同一脚本
> 在 96k/104k/110k/115k 四个不同档案上给出 59.4/70.1/67.2/97.6 tok/s，与上下文长度无关）。





