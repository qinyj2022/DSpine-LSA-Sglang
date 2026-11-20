# DSpine 网络结构完整说明（Qwen3.8-27B 版）

> 对象：`tencent/Qwen3.8-27B-DSpine-b16`（本机部署在 `/mnt/sda/<user>/dspine_draft`）
> 依据：论文 [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)、权重 `config.json` 与 `model.safetensors` 张量表（本文实测导出）、
> `sglang_dspine` 插件源码、`sglang/srt/models/dflash.py`（父类）。
> 标注 **【实测】** 的数字来自本文档配套工具在这台机器上的导出；其余来自上游代码与文档。

---

## 0. 一句话概括

DSpine 是一个 **2.03 B / 5 层** 的并行草稿模型：它复用目标模型的 token embedding 与 lm_head，
吃目标模型 5 个中间层的 hidden state 作为上下文，用 **"前缀码桥"（code bridge）** 把 token 之间的
因果依赖从"时间轴"搬到"网络深度轴"，从而**一次前向产出 16 个草稿 token**。

---

## 1. 总体结构

```
                    目标模型（Qwen3.8-27B，64 层）运行中
        ┌──────────────────────────────────────────────────────┐
        │  第 1 / 16 / 31 / 46 / 61 层 hidden      token embedding │
        └────────┬──────────────────────────────────┬──────────┘
                 │ [N, 5×5120]                      │ [16, 5120]
                 ▼                                  │
        ┌─────────────────┐                         │
        │ fc  25600 → 5120 │                         │   ← 输入融合（context stream）
        │ hidden_norm      │                         │
        └────────┬─────────┘                         │
                 │ 每层投影成 K/V                        │  ← 块内 token（query stream）
                 ▼                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │  Layer 0 ──(attn + MLP)──▶ 桥: read→message→shift→write     │
      │  Layer 1 ──(attn + MLP)──▶ 桥 ...                          │ 5 层
      │  Layer 2 ──(attn + MLP)──▶ 桥 ...                          │
      │  Layer 3 ──(attn + MLP)──▶ 桥 ...                          │
      │  Layer 4 ──(attn + MLP)──▶ 桥（末层必须带 norm 融合写入）     │
      └───────────────────────────┬────────────────────────────────┘
                                  │ [16, 5120] hidden
                                  ▼
                    第二轮 refine（serial_table, K=16）
                                  ▼
                        目标模型的 lm_head（共享，草稿不带）
                                  ▼
                          16 个草稿 token 的 logits
```

**两条数据流**

| 流 | 来源 | 作用 |
|---|---|---|
| 上下文流（context） | 目标模型 5 个 tap 层的 hidden → `fc` → `hidden_norm` | 每层投影成 K/V，让草稿"看到"真实上下文 |
| 块内流（query） | 目标模型 token embedding（16 个位置） | 16 个并行草稿位置本身 |

两者在每层自注意力处汇合：**16 个 query 对上下文 KV + 块内 KV 做因果注意力**。

---

## 2. 权重张量全表【实测：71 个张量，2,025,935,264 参数 ≈ 2.026 B，bf16 3.77 GiB】

### 2.1 顶层

| 张量 | 形状 | 参数量 | 作用 |
|---|---|---|---|
| `fc.weight` | [5120, 25600] | 131.07 M | 把 5 个 tap 层拼接（5×5120=25600）投影到 5120 |
| `hidden_norm.weight` | [5120] | 5.1 K | 融合后的 RMSNorm |
| `norm.weight` | [5120] | 5.1 K | 末层（norm 融合写入）用的最终归一化 |

### 2.2 `layers.{0..4}`（5 层，结构完全相同）

| 张量 | 形状 | 参数量 | 说明 |
|---|---|---|---|
| `self_attn.q_proj` | [4096, 5120] | 20.97 M | 32 头 × 128 |
| `self_attn.k_proj` | [1024, 5120] | 5.24 M | **GQA 32/8**，8 个 KV 头 |
| `self_attn.v_proj` | [1024, 5120] | 5.24 M | |
| `self_attn.o_proj` | [5120, 4096] | 20.97 M | |
| `self_attn.q_norm` / `k_norm` | [128] / [128] | 256 | QK-norm（Qwen3 风格） |
| `mlp.gate_proj` / `up_proj` | [17408, 5120] | 89.13 M ×2 | SwiGLU |
| `mlp.down_proj` | [5120, 17408] | 89.13 M | |
| `input_layernorm` / `post_attention_layernorm` | [5120] | 10.2 K | |

**每层 319.83 M × 5 = 1.599 B（占全部参数 79%）**

### 2.3 `selected_prefix_bridge.*` — DSpine 的核心

| 张量 | 形状 | 参数量 | 作用 |
|---|---|---|---|
| `codebook` | [248320, 1024] | 254.28 M | **词表级可学习码本**（248,320 = 词表大小），每个 token 一个 1024 维前缀码 |
| `read_weight` | [1024, 5120] | 5.24 M | 目标 hidden → 码空间（第 0 层读取） |
| `early_read_weights` | [4, 1024, 5120] | 20.97 M | 第 1–4 层的逐层读取 |
| `feature_write_weights` | [5, 5120, 512] | 13.11 M | 每层把消息写回 hidden（512 → 5120） |
| `prev.weight` | [512, 1024] | 0.52 M | 前驱码投影 |
| `predecessor_gate.weight` | [512, 1024] | 0.52 M | 前驱门控 |
| `gate.weight` / `gate.bias` | [512, 1024] / [512] | 0.52 M | 门控 |
| `out.weight` | [1024, 512] | 0.52 M | 消息 → 码 |
| `draft_block_causal_version` | `[]` | 0 | checkpoint 特性标记（标量） |
| `independent_write_version` | `[]` | 0 | 同上 |
| `terminal_disabled_version` | `[]` | 0 | 同上 |
| `last_write_refine_version` | `[]` | 0 | 同上 |

**桥接总参数 295.6 M（占 14.6%）**。注意它**不含** `embed_tokens` 与 `lm_head`——那两个由目标模型提供。

### 2.4 参数构成汇总

| 组成 | 参数量 | 占比 |
|---|---|---|
| 5 × decoder layer | 1.599 B | 78.9% |
| codebook | 254.3 M | 12.6% |
| fc（输入融合） | 131.1 M | 6.5% |
| 桥接读写适配器 | 41.4 M | 2.0% |
| 其它 norm | 15.4 K | ~0% |

---

## 3. 超参数全表（来自 `config.json`，逐项核对）

| 字段 | 值 | 说明 |
|---|---|---|
| `architectures` | `["DFlashDraftModel"]` | 磁盘上是 DFlash 架构名；插件按桥标记改判为 DSpine |
| `num_hidden_layers` | **5** | 草稿深度 |
| `hidden_size` | **5120** | 必须与目标模型一致 |
| `intermediate_size` | **17408** | MLP 中间维（与目标同宽） |
| `num_attention_heads` / `num_key_value_heads` | **32 / 8** | GQA |
| `head_dim` | 128 | |
| `layer_types` | 5 × `sliding_attention` | **全部是滑窗注意力** |
| `sliding_window` / `max_window_layers` | **8192 / 5** | 草稿只看最近 8192 token |
| `rope_theta` | 1e7 | `rope_type: default`，无缩放 |
| `rms_norm_eps` | 1e-6 | |
| `hidden_act` | silu | |
| `tie_word_embeddings` | **false** | 草稿不带 embedding，复用目标 |
| `vocab_size` / `draft_vocab_size` | **248320** | codebook 行数与之相等 |
| `block_size` | **16** | 每次并行起草 16 个位置 |
| `num_target_layers` | 64 | 目标层数（用于 tap 校验） |
| `max_position_embeddings` | 262144 | |

### 3.1 `dflash_config`（DSpine 专属标记）

| 键 | 值 | 作用 |
|---|---|---|
| `target_layer_ids` | **[1, 16, 31, 46, 61]** | 抽取哪 5 层目标 hidden（4 个是线性注意力层） |
| `mask_token_id` | **248070** | 并行填空的占位符（训练时新造，目标模型原生没有） |
| `draft_block_causal` | true | 块内因果注意力（必需） |
| `selected_prefix_code_bridge` | `{code_size: 1024, rank: 512, eps: 1e-6, kappa: 10.0}` | 码维 / rank / 归一化 eps / 门控锐度 |
| `selected_prefix_feature_layers` | [1, 2, 3, 4, 5] | 哪几层挂桥（全部） |
| `selected_prefix_feature_write_mode` | `independent` | 每层独立写入（必需） |
| `selected_prefix_feature_writes` | true | 必需 |
| `selected_prefix_terminal` | **false** | 插件只支持 terminal-disabled 桥 |
| `selected_prefix_decode_mode` | `serial` | 串行精化 |
| `selected_prefix_last_write_refine` | `{kind: exact_sender, head_topk: 16, version: 1, weight: 1.0}` | 末层二次精化 |
| `code_bridge_second_pass` | `{mode: serial_table, first_exact: 4, head_topk: 16}` | 第二轮：只重算末层写入，宽度 16 |
| **训练期元信息** | `selected_prefix_feature_gt_training: true`、`feature_teacher_layers: [1,2,3]`、`teacher_schedule: {kind: block_bernoulli, zero: 1/3, hold: 1/6, peak: 0.5}`、`context_fusion_checkpointing`、`decoder_checkpointing` | 训练配方，推理不参与计算 |

---

## 4. 前向流程（源码主线）

### 4.1 上下文融合（父类 `DFlashDraftModel`）

```python
# sglang/srt/models/dflash.py::project_target_hidden
target_hidden : [N, 5*5120]          # 5 个 tap 层拼接
proj  = fc(target_hidden)            # [N, 5120]
ctx   = hidden_norm(proj)            # 供每层投影成 K/V
```

维度校验是硬性的：`fc.in_features` 必须等于 `num_context_features × hidden_size`，
否则报 `DFLASH target_hidden feature dim mismatch`。

### 4.2 草稿主干（`DSpineDraftModel.forward`）

```python
hidden_states = input_embeds          # 目标模型的 token embedding（16 行）
residual, input_normed = None, False
for index, layer in enumerate(self.layers):          # 5 次
    hidden_states, residual = layer(positions, hidden_states,
                                    forward_batch, residual,
                                    input_normed=input_normed)
    input_normed = False
    if index in bridge.active_layers and hidden_states.numel():
        next_norm = self.layers[index+1].input_layernorm if index < last else self.norm
        stash = {} if index == last else None
        hidden_states, residual = bridge(hidden_states, residual, input_ids,
                                         index, next_norm, stash)
        input_normed = True          # 下一层的 input RMSNorm 已被融合掉
if not input_normed:
    raise RuntimeError("DSPINE's last layer must end with a (norm-fused) code write")
return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)
```

三个要点：

1. **每层之后都过桥**（`active_layers` = 全部 5 层）。
2. **norm 融合**：桥直接输出"已归一化"的 hidden，供下一层直接消费，省一次 RMSNorm。
3. **末层强制约束**：最后必须是一次"norm 融合的码写入"，否则运行期报错。

### 4.3 单层内部（`DSpineDecoderLayer`，继承 `DFlashDecoderLayer`）

```
x ──(input_layernorm，或由桥预归一化)──▶ self_attn ──(+residual)──▶ post_attention_layernorm ──▶ MLP
```

自注意力：**块内因果**（`draft_block_causal: true`）+ **8192 滑窗** + 对上下文 KV 的注意力。

---

## 5. 前缀码桥：因果如何沿深度传播

### 5.1 读取（read，每层开始）

```python
# _read_codes(hidden, residual, input_ids, layer)
raw, normed, rms = prepare_read(hidden, residual_or_hidden, eps)   # 融合残差 + RMSNorm
projected = F.linear(normed, layer_read_weight(layer))             # 5120 → 1024
anchors   = input_ids.view(-1, 16)[:, 0]                           # 每块第 0 行 = 锚点 token
codes     = normalize_codes(projected, codebook, anchors, eps)     # [m, 1024]
```

第 `layer` 层用 `read_weight`（layer 0）或 `early_read_weights[layer-1]`（layer 1–4）。

**含义**：每个位置的 1024 维"前缀码" = 目标 hidden 的投影 + **锚点 token 在码本里的可学码**。
码本是词表级的（248,320 行），所以它等价于"每个 token 有一个学出来的 1024 维语义码"。

### 5.2 写入与相邻注入（write，每层结束）

```python
# _forward_fused(...)
packed  = F.linear(codes, packed_weight)                    # 码 → 512 维（含 prev / 前驱门控 / gate）
message = feature_message(packed, gate.bias, r=512, t=16)   # ← 在块内按相邻位置做位移
delta   = F.linear(message, feature_write_weights[layer])   # 512 → 5120
output  = raw + delta                                       # 加到残差流上
(normed, output) = feature_finish(output, rms, next_norm)   # 提前归一化，供下一层
```

`feature_message` 的关键参数是 **`t = block_size = 16`**：内核按块内位置做邻居位移，
让位置 `p` 的消息进入位置 `p+1` 的状态。

### 5.3 为什么这叫"按深度条件化"

| 层 | 位置 p 可获得的因果信息 |
|---|---|
| 0 | 只有自身 + 锚点码 |
| 1 | p−1 的初猜（第 0 层写入的消息） |
| 2 | p−1 经过 1 层修正后的猜测 |
| … | … |
| 4 | p−1 的深度精化结果 |

于是"第 p 个 token 依赖第 p−1 个 token"这条链，不再需要 16 次串行前向，
而是被 **5 层深度**分摊消化——这是并行起草能成立的根本原因。

---

## 6. 第二轮精化（`DSpineDraftSampler`）

主干输出后还有一步串行表精化：

```json
"code_bridge_second_pass": {"mode": "serial_table", "first_exact": 4, "head_topk": 16},
"selected_prefix_last_write_refine": {"kind": "exact_sender", "head_topk": 16, "weight": 1.0, "version": 1}
```

- **只重算末层写入**（`first_exact = num_layers − 1 = 4`），不是全量重跑；
- 精化宽度 **K = 16**（`head_topk`），即对 16 个候选做"精确发送方"打分；
- 权重 1.0，无额外缩放。

插件对这段有硬校验：`mode` 必须是 `serial_table`、`first_exact` 必须等于 `num_layers−1`，
否则直接拒绝加载。

---

## 7. 验证与动态宽度策略（`sglang_dspine/verification/`）

| 模块 | 文档字符串（原文） | 作用 |
|---|---|---|
| `policy.py` | "Measured verification widths and token budgets, indexed by actual batch size" | 按**实际 batch size** 查表决定验证宽度与 token 预算 |
| `sps.py` | "DSpark prefix allocation within a measured, per-batch token budget" | 在预算内做 **DSpark 式前缀分配**（每条请求分多少草稿） |
| `calibration.py` | "Post-hoc maps from draft confidence to conditional acceptance probability" | 把草稿置信度事后标定成条件接受率（深度 0 不含已知锚点） |
| `ragged.py` | — | 不规则 batch 的验证编排 |
| `caps.py` | — | 采样随机流/种子等容量约束检查 |

这套机制解释了为什么实测 accept 会随内容波动：**验证宽度不是固定 16，而是按批内情况动态分配的**。

---

## 8. 与目标模型的耦合点

| 耦合 | 具体表现 | 后果 |
|---|---|---|
| hidden 空间共享 | 草稿 `hidden_size = 5120`、MLP 17408 与目标一致 | 不能换不同宽度的目标模型 |
| embedding 共享 | 草稿无 `embed_tokens`，`has_embed_tokens` 被显式拒绝 | 必须由目标 embedding 提供输入 |
| lm_head 共享 | 草稿无 lm_head，logits 由目标 `lm_head` 计算 | 词表必须对齐（248,320） |
| 特征抽取点 | `target_layer_ids = [1,16,31,46,61]` | 目标层数/布局一变就失效 |
| 占位符 | `mask_token_id = 248070` 训练时新造 | 目标模型本身没有填空能力 |
| 线性链约束 | 桥按块内相邻位置位移 | 只支持密集线性块，不支持树状/多分支 |

---

## 9. sglang 集成方式

`plugin.py` 注册一个新的投机算法 `DSPINE`，monkey-patch 五个内部点：

| hook | 目标 |
|---|---|
| `handle_speculative_decoding` | `sglang.srt.arg_groups.speculative_hook.handle_speculative_decoding` |
| `spec_need_hidden_states` | `sglang.srt.speculative.spec_utils.spec_need_hidden_states`（DSpine 恒为 False，改为走 aux hidden） |
| `decode_cuda_graph_runner_cls` | `ModelRunner._decode_cuda_graph_runner_cls` |
| `config_draft_model` | `sglang.srt.configs.model_config.ModelConfig._config_draft_model`（按桥标记改写 architecture） |
| `dflash_sampling_verify` | `sglang.srt.speculative.dflash_worker_v2.compute_dflash_sampling_correct_drafts_and_bonus` |

类继承关系：`DSpineDraftModel(DFlashDraftModel)`、`DSpineDecoderLayer(DFlashDecoderLayer)`
——**与 DFlash/DFlash2 共用同一条 sglang 投机管线**，差异集中在码桥与深度注入。

---

## 10. 硬约束清单

| 约束 | 错误信息（原文） |
|---|---|
| 必须 CUDA | `DSPINE speculative decoding requires CUDA` |
| **tp_size 必须 1** | `DSPINE speculative decoding supports tp_size == 1 only` |
| 不支持草稿窗口参数 | `DSPINE does not support --speculative-draft-window-size` |
| 不支持 PD 分离 | `DSPINE does not support prefill/decode disaggregation` |
| 不支持 FP4 KV | `its verify commits a prefix of a dense draft block...validated with the default KV cache only` |
| ReplaySSM 需 pp=1 | `DSPine ReplaySSM port requires pp_size == 1` |
| ReplaySSM 需 radix | `DSPINE prefix cache requires radix caching enabled` |
| ReplaySSM 需 extra_buffer | `DSPINE ReplaySSM prefix caching requires extra_buffer checkpoints` |
| ReplaySSM 需 fp32 SSM | `DSPINE ReplaySSM requires --mamba-ssm-dtype float32` |
| ReplaySSM 需 triton 线性注意力 | `DSPINE ReplaySSM requires --linear-attn-backend triton` |
| 草稿不支持量化 | `The DSPINE draft does not support quantized weights` |
| 不支持分组卷积草稿 | `DSpine checkpoints do not use grouped draft convolutions` |
| 桥必须是 terminal-disabled | `DSPINE supports terminal-disabled code bridges only` |
| 必须是独立写入 | `DSPINE requires independent learned writes` |
| 必须块内因果 | `DSPINE requires causal draft attention (draft_block_causal)` |
| 末层必须 norm 融合写入 | `DSPINE's last layer must end with a (norm-fused) code write` |

---

## 11. 可调参数：block 宽度 2..16【重要】

`DSpineDraftModel.set_block_size()` 允许运行时把 block 宽度设成 **2 到 16**（训练宽度为上限）：

```python
if not 2 <= t <= self.trained_block_size:
    raise ValueError(f"DSPINE serves block sizes 2..{self.trained_block_size}...")
```

含义与代价：

| block 宽度 | 每步验证行数 | 单步耗时 | accept（每步出几个 token） |
|---|---|---|---|
| 16（当前） | 16 | 基准 | 高（实测 5.0–5.4） |
| 8 | 8 | 更低 | 更低 |
| 2 | 2 | 最低 | 接近提示查表 |

**这是一个尚未实测的调优点**：步耗时随行数近似线性（本机测过 ~0.76 ms/行），
如果 accept 的下降慢于行数的下降，窄块可能更划算。配置入口是 `--speculative-num-draft-tokens`。

---

## 12. 本机实测行为

| 观测 | 数值 | 说明 |
|---|---|---|
| 权重大小 | 3.77 GiB（bf16，4,051,831,880 字节） | 加载耗时 0.83 s |
| 每步固定开销 | 短上下文步耗时 ~50 ms（含目标 51 GB 权重流） | 与草稿 3.77 GiB 逐位无关上下文 |
| 长上下文表现 | 118k 步 58.9 ms（flashinfer）/ 54.3 ms（LSA） | 草稿 8192 滑窗使其不随上下文增长 |
| accept（本机负载） | 5.0 – 5.4 | 官方 GSM8K 7.20 / HumanEval 5.71 / MT-Bench 4.48 |
| 吞吐 | 短上下文均值 130–135 tok/s；118k 61–73 tok/s | 五类负载 / 399-token 长输出 |
| 相对 dense 加速 | **5.28×**（短上下文） | dense 均值 24.7 tok/s |
| 显存 | 草稿 3.92 GB + 目标 51.05 GB + Mamba 9.53 GB | 三者共同决定 KV 池上限 |

---

## 13. 与其它草稿方案对比

| 方案 | 草稿规模 | 因果建模位置 | 需要训练 | 块宽 | 备注 |
|---|---|---|---|---|---|
| **DSpine** | 5 层 / 2.03 B | **网络深度**（码桥位移） | 是 | 2–16（训练 16） | 本机 accept 5.0–5.4 |
| DFlash / DFlash2 | 多层 + 锚点行 | 锚点行 + 并行 | 是 | 8（官方） | 同一管线，无码桥 |
| DSpark | 稠密并行块 | 并行 | 是 | — | DSpine 的验证策略参考它 |
| EAGLE-3 | 单/多层头 | 自回归时间轴 | 是 | 链长 | 需逐 token 串行 |
| Medusa | 多头 | 各头独立并行 | 是 | 头数 | 头间无因果 |
| prompt-lookup | 无模型 | 查表 | **否** | — | 本机实测 accept 1.50 |

---

## 14. 复现方法

```bash
# 权重骨架（不加载张量，直接读 safetensors 头）
python tools/st_shape.py dspine_draft/model.safetensors

# 判定任意 checkpoint 是否为 DSpine 草稿（看 selected_prefix_code_bridge）
python tools/draft_probe.py dspine_draft/config.json

# tap 层是否落在递归（线性注意力）层上
python tools/layer_map.py dspine_draft/config.json dspine_train/model_view/config.json

# Mamba/SSM 显存账
python tools/mamba_math.py dspine_train/model_view/config.json 64

# 目标模型是否具备 FIM/填空 token（决定能否零训练做并行草稿）
python tools/fim_probe.py dspine_train/model_view
```

以上工具位于本工作区 `tools/`，远程副本在 `/mnt/sda/<user>/` 同名文件。

---

## 15. 关键结论

1. **草稿很轻，但不可省**：2.03 B / 3.77 GiB，约占目标模型参数的 7.5%，换来 5.28× 吞吐。
2. **DSpine 的新意全在码桥**：5 层普通 Qwen3 层之上，靠 1024 维词表码本 + 每层 512 维消息位移，
   把因果从时间轴搬到深度轴。去掉桥就是 DFlash。
3. **它与目标模型强耦合**：hidden 宽度、词表、embedding、lm_head、5 个 tap 层、占位符 token，
   任一项变化都需要重新训练草稿。
4. **块宽度是可调而非固定**：2–16 之间可选，是当前尚未验证的性能/接受率权衡点。
5. **草稿自带 8192 滑窗**：这是长上下文下草稿开销不涨的结构原因，也解释了为什么
   `--speculative-draft-window-size`（另一回事）被插件拒绝。
