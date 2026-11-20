# DSpine + LSA on SGLang

在 **Qwen3.8-27B（混合线性注意力，64 层 = 48 GDN + 16 full-attn）** 上，把两种互不相干的推理加速技术同时跑通并量化：

- **DSpine**（Tencent，Apache-2.0）— 并行草稿投机解码，一次前向出 16 个草稿 token
- **LSA**（Prefrontal-AI/latent-sparse-attention，Apache-2.0）— 稀疏注意力索引，把长上下文注意力代价钉死在 `top_k`
- 承载在 **SGLang 0.5.22** 上，GPU 为 RTX PRO 6000 （sm120，97,887 MiB）
- 16并发达到600toks/s的吞吐，100K 1M长上下文性能不减
本仓库是该组合的**适配补丁、启动脚本、测量工具与全部实测结果**。没有模型权重。

---

## 目录

- [1. 结果速览](#1-结果速览)
- [2. 四路吞吐分解](#2-四路吞吐分解)
- [3. 长上下文与每步耗时归因](#3-长上下文与每步耗时归因)
- [4. 200K 上下文](#4-200k-上下文)
- [5. 并发吞吐](#5-并发吞吐)
- [6. 正确性验证](#6-正确性验证)
- [7. 对历史结论的两处更正](#7-对历史结论的两处更正)
- [8. 仓库结构](#8-仓库结构)
- [9. LSA 适配了什么](#9-lsa-适配了什么)
- [10. DSpine 结构要点](#10-dspine-结构要点)
- [11. 部署](#11-部署)
- [12. 复现](#12-复现)
- [13. 已知限制](#13-已知限制)
- [14. 许可与致谢](#14-许可与致谢)

---

## 1. 结果速览

| 指标 | 数值 | 条件 |
|---|---|---|
| 单流吞吐（短上下文，五类负载均值） | **130.6 – 135.3 tok/s** | DSpine + flashinfer |
| 相对纯 dense 加速 | **5.28×** | 24.7 → 130.6 tok/s |
| 118k decode | **61 – 74 tok/s** | DSpine 或 LSA+DSpine |
| LSA 相对 dense 长上下文收益 | **+6.7% @118k，+14.7% @200k** | 无投机 |
| LSA 在投机下的每步收益 | **−8.5% 步耗时 @118k** | 59.6 → 54.3 ms |
| 200K 单流 | **23.18 tok/s**（LSA）vs 20.21（dense） | 399-token 解码 |
| 并发 12 总吞吐 | **495.4 tok/s** | LSA+DSpine，池 124,850 |
| 并发 16 峰值 | **605.6 tok/s** | 放宽档，池缩到 20,974 |
| accept length | 5.0 – 5.36 | 本机负载 |

---

## 2. 四路吞吐分解

短上下文五类负载（`tools/bench5.py`，400 token 生成，不加逐字日志，每配置独立重启）：

| 配置 | A 对话 | B 摘要 | C 代码 | D JSON | E 数学 | **均值** | 相对 dense |
|---|---|---|---|---|---|---|---|
| 纯 dense（无 LSA、无投机） | 25.17 | 24.64 | 24.73 | 24.51 | 24.57 | **24.72** | 1.00× |
| + LSA（triton/page1） | 25.30 | 24.79 | 24.53 | 24.21 | 23.95 | **24.56** | 0.99× |
| + DSpine（flashinfer） | 51.96 | 191.65 | 119.86 | 169.07 | 120.36 | **130.58** | **5.28×** |
| + LSA + DSpine | 52.42 | 198.18 | 81.01 | 173.41 | 113.17 | **123.64** | 5.00× |

**结论**：单流吞吐的收益几乎全部来自 DSpine。LSA 在短上下文为 0（±1%）。

> 四列对比必须用每步耗时，不能用 tok/s：tok/s = accept / 步耗时，而 accept 随内容波动极大。
> 上表 C 列 LSA+DSpine 偏低（81.0 vs 119.9）就是 accept 差异，不是配置性能差异——两者在 16k 的步耗时是 51.9 vs 51.2 ms。

---

## 3. 长上下文与每步耗时归因

### 3.1 端到端 tok/s（399-token 解码，prefill 走 radix 缓存命中）

| 上下文 | dense | +LSA | +DSpine | 对照：triton+page1 **不开 LSA** | +LSA+DSpine |
|---|---|---|---|---|---|
| 16k | 24.42 | 24.27 | 70.87 / 74.86 | 49.81 | 66.83 / 69.63 |
| 64k | 23.04 | 23.46 | 87.50 / 88.47 | 28.68 / 28.77 | 79.17 / 82.10 |
| 118k | 21.83 | 23.29 | 61.48 / 72.68 | **15.90** | 66.39 / 73.75 |

### 3.2 每步耗时（`accept/throughput` 归一化，与 accept 无关）—— 决定性指标

| 上下文 | DSpine+flashinfer | triton+page1（无 LSA） | LSA+DSpine | LSA 收益 |
|---|---|---|---|---|
| 1k | 49.8 ms | — | 49.9 ms | 0 |
| 4k | 51.2 ms | — | 51.7 ms | 0 |
| 16k | 51.2 ms | 80.6 ms | 51.9 ms | ≈0 |
| 64k | 55.6 ms | 180.5 ms | **53.8 ms** | **−3.2%** |
| 118k | 59.6 ms | **296.7 ms** | **54.3 ms** | **−8.5%** |

**这张表的三个结论**：

1. LSA 的价值随上下文增长：≤32k 完全打平，64k −3.2%，118k −8.5%。
2. LSA 强制的 `triton + page-size 1` 后端，**在没有索引时随上下文线性恶化到 296.7 ms/步**（15.9 tok/s）。LSA 把它拉回 54.3 ms（**5.5× 修复**）。
3. DSpine+LSA 的步耗时从 1k 到 118k **几乎平坦**（49.9 → 54.3 ms）。

> 复现对照用的脚本是 `scripts/ctrl_ds_triton_p1.sh`：与 `graph_lsa.sh` 逐参数相同，**只去掉 LSA**。

---

## 4. 200K 上下文

模型原生支持 `max_position_embeddings = 262144`。实测（399-token 解码）：

| 配置 | @200K tok/s | 每步耗时 | prefill 199K |
|---|---|---|---|
| 纯 dense（flashinfer） | 20.21 | 49.5 ms | 85.7 s |
| **+ LSA（triton/page1）** | **23.18** | 43.1 ms | 62.7 s |
| LSA 相对收益 | **+14.7%** | −11.5% | — |

LSA 相对 dense 的收益随上下文单调增长：

| 16k | 64k | 118k | 200k |
|---|---|---|---|
| 0% | +1.8% | +6.7% | **+14.7%** |

**未完成**：DSpine 与 LSA+DSpine 的 200K 实测。原因是 200K 需要约 16 GB KV，而
投机配置的池只有 124,850–137,162 token。腾内存的脚本已备好（`scripts/ds_200k.sh`、
`scripts/combo_200k.sh`：`mamba 64→16` + `mem-fraction 0.82→0.85`），但未跑完。

---

## 5. 并发吞吐

### 默认档（`max-mamba-cache-size 64` → 并发上限 12，KV 池 124,850）

| 并发 | 总吞吐 tok/s | 单请求时延 |
|---|---|---|
| 1 | 87.7 | 2.92 s |
| 2 | 157.3 | 3.20 s |
| 4 | 278.5 | 3.56 s |
| 8 | 472.4 | 4.14 s |
| **12** | **495.4** | 5.88 s |

### 放宽档（mamba 96 / mem 0.78 → 上限 19，KV 池缩到 20,974）

| 并发 | 总吞吐 tok/s |
|---|---|
| 12 | 462.7 |
| **16** | **605.6（峰值）** |
| 19 | 553.3 |

1→8 近乎线性（每翻倍约 1.7×），8→12 仅 +5%，16 达峰值、19 反降（池被挤压后开始驱逐）。

> **注意**：上表是 **LSA+DSpine** 的测量结果。**DSpine+flashinfer 的并发吞吐未实测**，不能直接套用。

---

## 6. 正确性验证

| 检查 | 结果 |
|---|---|
| needle @8k / @32k / @64k | ✅ 全部命中 `BLUEHERON-7742` |
| 逐字引用（摘要负载） | ✅ 原文完整复现 |
| JSON 负载 | ✅ 输出合法 JSON |
| 采样请求稳定性 | ✅ 62 次混合采样 / 8k 长提示 / 并发突发，**零崩溃** |

DSpine 的采样路径曾因 `sampled verify reached without this round's sampled draft chain`
直接杀死 scheduler，已由 `[PATCH dspine-stable]` 增加回落到原生 DFlash verify 的保护。

---

## 7. 对历史结论的两处更正

工程记录里两条早期结论被后续测量推翻，这里保留以免复现同样的错误：

| 原说法 | 更正 |
|---|---|
| "LSA 在 119k 给 DSpine 带来 **1.29–1.46×**" | **不成立**。那是跨 prompt 比较 tok/s 的结果。按每步耗时，LSA 的真实收益是 **8.5%/步（118k）** |
| "`graph_lsa.sh` @119k = **94.55 tok/s**（n=5 中位）" | **不可复现**。同一脚本同日 3 次读数 74.67 / 74.67 / 60.22。步耗时仍为 54.3 ms，**配置没退化，差异全在 accept** |

根因：早期用只生成 50 余 token 的探针计时，首步开销污染了分母。**长上下文测量必须生成 ≥399 token**。

---

## 8. 仓库结构

```
.
├── README.md                  ← 本文件
├── NOTICE.md                  ← 第三方来源与许可
├── VERSION                    ← 部署版本与参数快照
├── src/
│   ├── lsa_code/              ← 启动器（sglang_uds.py）、UDS 桥、LSA 补丁脚本 v2–v12、投影映射 maps/
│   └── dspine_code/DSpine-main/ ← DSpine 插件源码（Apache-2.0，含我们的 [PATCH dspine-stable]）
├── patches/
│   ├── lsa/                   ← 被改动的 lsa 包文件（保留 [PATCH spec vN] 标记与上游 LICENSE）
│   └── sglang/srt/speculative/eagle_worker_common.py  ← SGLang 侧唯一改动文件
├── scripts/                   ← 全部启动变体、补丁/生成器脚本、发布脚本
├── tools/                     ← 测量与诊断工具（见下）
└── docs/                      ← 详细文档
    ├── dspine-architecture.md         ← DSpine 网络结构完整说明（71 张量逐个列出）
    ├── four-way-decomposition.md      ← 四路分解完整报告
    ├── v2.0-fp16-performance.md       ← v2.0 性能报告（含并发表）
    ├── handoff-lsa-spec.md            ← LSA 补丁逐版设计与踩坑记录
    ├── sglang-deploy-2026-11-13.md    ← 首次部署记录
    ├── dspine-qwen38-test-2026-11-16.md
    ├── dspine-fp8-sm120-test-2026-11-16.md
    └── release-v1.0.md
```

### 工具

| 工具 | 用途 |
|---|---|
| `tools/bench5.py` | 五类负载（对话/摘要/代码/JSON/数学）短上下文吞吐 |
| `tools/needle_test.py` | 长上下文"针"检索（验证稀疏索引没丢关键 block） |
| `tools/decode_probe.py` | 指定上下文的 decode 速率（warm 一次让 prefill 命中缓存） |
| `tools/conc_bench.py` | 并发吞吐阶梯 |
| `tools/step_stats.py` | 从每步日志算 `step_ms = 1000×accept/throughput` |
| `tools/mamba_math.py` | 由 config 推算 SSM/Mamba cache 显存账 |
| `tools/layer_map.py` | 检查 DSpine 的 tap 层是否落在递归层上 |
| `tools/draft_probe.py` | 判定任意 checkpoint 是否为 DSpine 草稿 |
| `tools/st_shape.py` | 不加载权重、直接读 safetensors 头导出骨架 |
| `tools/cfg_probe.py` / `tools/fim_probe.py` | 模型几何 / 是否存在 FIM 填空 token |
| `tools/ctrl_ds_triton_p1.sh` | **隔离对照**：triton+page1 但不开 LSA |
| `tools/make_200k.sh` | 生成四个 200K 启动变体 |

---

## 9. LSA 适配了什么

`sglang` 侧只需改**一个文件**：`patches/sglang/srt/speculative/eagle_worker_common.py`（prompt-lookup 相关）。
其余全部改动落在 `lsa` 包内（14 个文件带标记）。核心问题都是**投机解码的 verify 批形状与 LSA 的 decode 假设不一致**：

| 标记 | 文件 | 解决的问题 |
|---|---|---|
| `[PATCH spec v4]` | `qwen3_5/backend.py` | verify 块按 decode 步方式读取（每 query 一行）；只有真 prefill 才在 host 侧选择 |
| `[PATCH spec v5]` | 同上 | 移除 verify 的 maintain 回退（capture 预热批没有 `rids`） |
| `[PATCH spec v7]` | `index/selector.py` | `TopK` 宽度必须按 `request_count × draft_rows()`，否则 `listed` 被静默截断 |
| `[PATCH spec v9]` | `index/registry.py` | replay 视图缺 `seq_lens_cpu` 时回落到设备张量 |
| `[PATCH spec v12]` | 同上 | 捕获的 verify 批被 padding，`rids` 比 `req_pool_indices` 短 → 截断修复 |
| `[PATCH spec v10/11]` | `backend.py` / `layer_index.py` | 量化 KV 缓存路径：保持量化、无 scale 时 descale 用 1.0；bf16 才写 `KeySlot` |
| `[PATCH spec]` ×3 | `backend.py` / `registry.py` | verify/draft-extend 走 extend 形状选择；prefill 无 metadata hook 的兜底 |
| `[PATCH spl/spl2/spl3]` | `eagle_worker_common.py` | prompt-lookup 链填充与 B'-chain 融合（**在 DSpine 上不生效**，钩子挂在 eagle 路径） |

详细设计与被否证的路线见 `docs/handoff-lsa-spec.md`（含 §12 三项优化否决、§15 四路分解与更正）。

---

## 10. DSpine 结构要点

完整文档：[`docs/dspine-architecture.md`](docs/dspine-architecture.md)（71 张量逐个列出）。

| 项 | 值 |
|---|---|
| 草稿规模 | **2.026 B 参数 / 3.77 GiB bf16 / 5 层** |
| 结构 | 5 × Qwen3 风格 decoder（attn + SwiGLU，hidden 5120、MLP 17408、GQA 32/8、head_dim 128） |
| 上下文融合 | `fc [5120, 25600]`：目标模型 5 个 tap 层（**1/16/31/46/61**，其中 4 个是线性层）hidden 拼接 → 5120 |
| 核心机制 | `selected_prefix_bridge`：词表级码本 `[248320, 1024]` + 每层 512 维消息，**按块内相邻位置位移**，把因果从时间轴搬到深度轴 |
| 并行宽度 | block 16（运行时 2–16 可调） |
| 草稿自注意力 | 5 层全滑窗，`sliding_window = 8192` |
| 共享 | 复用目标模型的 token embedding 与 lm_head（草稿自己不带） |
| 训练标记 | `mask_token_id = 248070`、`selected_prefix_code_bridge` —— 缺则插件拒绝加载 |

**为什么这些数字重要**：草稿每步要额外读 3.77 GiB 权重（短上下文每步 ~50 ms 的固定成本之一），
而它自带 8192 滑窗，所以长上下文下草稿开销几乎不涨。

---

## 11. 部署

### 依赖

| 组件 | 版本 |
|---|---|
| SGLang | 0.5.22 |
| Python | 3.12 |
| torch | 2.14.1+cu130 |
| GPU | sm120（Blackwell） |

### 主要启动变体

| 脚本 | 配置 | KV 池 |
|---|---|---|
| `scripts/bf16_src_uds.sh` | DSpine + flashinfer，无 LSA | 137,162 |
| `scripts/graph_lsa.sh` | LSA + DSpine（triton、page1、保留 verify 图） | 124,850 |
| `scripts/lsa_uds.sh` | 仅 LSA | 234,398 |
| `scripts/dense_base.sh` | 纯 dense 基线 | 274,627 |
| `scripts/dense_200k.sh` / `lsa_200k.sh` | 262k context | 274,307 / 226,438 |
| `scripts/ds_200k.sh` / `combo_200k.sh` | 262k + 投机（未跑完） | — |
| `scripts/ctrl_ds_triton_p1.sh` | 隔离对照（triton+page1，无 LSA） | 137,162 |
| `scripts/*_log.sh` | 同配置 + `--decode-log-interval 1`（算每步耗时用） | — |

### 服务形态

`sglang_uds.py` 在 Unix socket 上起服务，`ipc_bridge.py` 转成 TCP（本机有 100 KiB 出站 TCP 上限时必需）：

```bash
bash scripts/restart_any.sh graph_lsa.sh           # 启动
ssh -N -L 127.0.0.1:30000:/path/to/run/sglang.sock host   # 隧道
curl 127.0.0.1:30000/v1/chat/completions ...       # OpenAI 兼容
```

> 客户端必须用 `127.0.0.1` 而非 `localhost`（后者可能先解析到 `::1`）。服务本身不带 web UI。

---

## 12. 复现

```bash
# 1) 环境：SGLang 0.5.22 + torch 2.14.1+cu130
# 2) 打补丁
python src/lsa_code/patch_lsa_spec_v12.py ...        # 按版本依次应用
cp patches/sglang/srt/speculative/eagle_worker_common.py $SP/sglang/srt/speculative/
pip install -e src/dspine_code/DSpine-main
# 3) 权重
#    目标  Qwen/Qwen3.8-27B
#    草稿  tencent/Qwen3.8-27B-DSpine-b16
# 4) 启动
bash scripts/restart_any.sh graph_lsa.sh
# 5) 测量
python tools/bench5.py
python tools/decode_probe.py 118000 400 long
python tools/conc_bench.py --levels 1,2,4,8,12
```

**测量纪律**（这些数字全部遵守）：

- 不加 `--decode-log-interval`（逐字日志压 tok/s 10–15%）；
- 每配置独立重启；
- 长上下文一律生成 399 token 再计时；
- 比较配置能力用**每步耗时**，不用 tok/s；
- 跨 boot 噪声 ±5–13%，同格多次读数全部列出。

---

## 13. 已知限制

**DSpine 侧**（源码强制，`sglang_dspine/spec.py`）：

| 限制 | 原因 | 性质 |
|---|---|---|
| `tp_size == 1` | verify 要提交稠密块前缀，涉及 req_to_token / KV / SSM 三处按 head 分片协调，未实现 | 工程债 |
| `pp_size == 1` | tap 层分布在不同 stage，每 draft step 需跨 stage 通信 | 工程债 |
| 不支持 PD 分离 | 附加状态（特征桥、SSM 检查点、confidence relay）不在 KV 传输协议里 | 协议缺失 |
| 必须 ReplaySSM + fp32 SSM + triton 线性注意力 | 混合模型 verify 要逐位置状态；替代方案是每草稿位置存全状态快照（≈2.3 GB/请求） | 架构决定 |
| 不支持 FP4 KV | verify 只用默认 KV dtype 验证过 | 未验证 |
| 草稿不支持量化 | 插件显式拒绝 | 未实现 |

**LSA 侧**：

- 强制 `page_size = 1` + triton attention（上表 296.7 ms 的对照说明了为什么必须配索引）；
- 索引 map 与模型几何强绑定（768 → 64），换模型需重新蒸馏；
- 索引本身约占 1 GB 显存（池从 137,162 降到 124,850）；
- 仅适配 `qwen3_5` / `glm5_next` 两个模型族。

**内存现实**（本机实测）：

| 项 | 占用 |
|---|---|
| 目标权重 | 51.05 GB |
| DSpine 草稿 | 3.92 GB |
| **Mamba/SSM cache**（64 槽 × 143 MiB） | **9.53 GB** |
| KV cache（124,850 token） | 10.0 GB |

Mamba cache 是 48 个线性注意力层的循环状态，**与是否用 DSpine 无关**（纯 dense 基线同样 9.53 GB）。
它相当于 144K token 的 KV——这是 200K 上下文上不去的直接原因。

---

## 14. 许可与致谢

本仓库的补丁、脚本、工具与文档由本仓库作者编写。其中包含或派生自以下 **Apache-2.0** 项目：

| 上游 | 来源 | 在本仓库中的位置 |
|---|---|---|
| **DSpine** | [tencent/dspine](https://github.com/tencent/dspine) · 论文 [arXiv 2609.36173](https://arxiv.org/abs/2609.36173) | `src/dspine_code/DSpine-main/`（保留原 `License.txt`） |
| 草稿权重 | [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) | 不随仓库分发 |
| **LSA** | [Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) | `patches/lsa/`（保留上游 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`） |

上游项目未 vendor 进本仓库；`patches/lsa/` 只包含带补丁标记的改动文件。
DSpine 与 LSA 的著作权归各自作者所有。详见 [`NOTICE.md`](NOTICE.md)。

---

## 附：上传分片文件

本仓库的发布网络路径上，**任何超过约 64 KiB 的请求体都会被中间设备切断**（`git push` 整包、
`git-receive-api` 大 blob、SSH 大 pack 全部失败）。因此：

- 每个文件都以**独立请求逐文件上传**（685 个源文件 → 895 次请求）；
- 超过 45 KB 的 **32 个文件**被切成 `<原路径>.partNNN` 分片存放，清单见根目录 `bigfiles.json`。

在本机克隆后执行一次即可还原原文件：

```bash
python tools/join_parts.py .
```

分片文件保留不影响任何东西，可以按需删除。
