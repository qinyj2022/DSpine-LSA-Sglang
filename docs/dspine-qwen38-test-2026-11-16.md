# DSpine 在 Qwen3.8-27B 上的实测报告

日期：2026-11-16
主机：`<gpu-host>`（`<gpu-host>`）
GPU：NVIDIA RTX PRO 6000 Blackwell Max-Q，**SM 12.0 (sm_120)**
工作目录：`/mnt/sda/<user>`

## 结论

**DSpine 在本机无法运行 Qwen3.8-27B。** 不是配置错误、不是依赖缺失，而是**GPU 架构硬不兼容**：

```
DSpine 源码：BACKENDS = frozenset(("fa3",))        ← 只接受 FlashAttention 3
SGLang FA3：  requires SM>=80 and SM<=90           ← 仅 A100 / H100
本机 GPU：    SM 12.0 (Blackwell)                  ← 不在范围内
```

三条约束形成死锁，无配置解法。

## 已完成的部分（全部成功）

| 环节 | 状态 | 证据 |
|---|---|---|
| 名称核实 | ✅ | `Tencent/DSpine`=200；`Tencent/BSPine`=404。**BSPine 不存在，是 DSpine** |
| 代码下载 | ✅ | `codeload` tar 包 1.2M，解压 `/mnt/sda/<user>/dspine_code/DSpine-main`（49 个 py 文件） |
| 草稿模型 | ✅ | `model.safetensors` = **4,051,831,880 B**，与 HF 元数据逐字节一致 |
| SGLang 0.5.22 | ✅ | 独立环境 `sglang522/`，16G，克隆自含 `torch 2.14.1+cu130` 的 conda env |
| DSpine 插件 | ✅ | `sglang-dspine 0.2.0`，entry point `dspine -> sglang_dspine.plugin:register` |
| DSPINE 算法注册 | ✅ | `CustomSpecAlgo('DSPINE')` |
| 草稿模型架构映射 | ✅ | `DFlashDraftModel` → `DSpineDraftModel`；bridge 校验通过 |
| **草稿模型加载** | ✅ | `DSPINE ready: block_size=16 head_topk=16 bridge=fused`（3.92 GB） |
| **主模型加载** | ✅ | `type=Qwen3_5ForConditionalGeneration, mem usage=51.05 GB` |
| **KV Cache 分配** | ✅ | `132,585 tokens, bfloat16, K/V 4.05+1.26 GB` |

**技术上 DSpine 已经完全跑通到 KV cache 分配完成，只差最后的 attention backend 初始化。**

## 实测排除的四个障碍（按遇到顺序）

### 障碍 1：NVFP4 模型的 lm_head 不兼容

```
ValueError: DSPINE needs an unquantized row-contiguous BF16 target lm_head
```

源码检查（`sglang_dspine/runtime/worker.py:529`）：
```python
if (weight is None or weight.dtype != torch.bfloat16
    or weight.dim() != 2 or weight.stride(1) != 1):
    raise ValueError(...)
```

**原因**：NVFP4 和 FP8 变体的 `lm_head` 都量化为 FP8（`re:.*lm_head` 在 group_0 targets 里）。

**解决**：改用 BF16 原始模型 `/mnt/sda/models/Qwen3.8-27B`（52G，`quantization_config: NO`）。

### 障碍 2：mamba cache 吃掉 37.6 GB

`--max-mamba-cache-size 256`（DSpine 官方建议值）导致：
```
[MEMDBG]  avail=34.40  slack=13.58  mm=0.10  rest=20.72
[MEMDBG2] mamba before=20.72  after=-16.91     ← 消耗 37.6 GB
```

**原因**：Qwen3.8-27B 是混合架构（48 层 Gated DeltaNet + 16 层 full attention），`mambaish_config` 非空，触发 `_handle_max_mamba_cache`。

**解决**：`--max-mamba-cache-size 64` → `after=10.62`，KV cache 成功分配。

### 障碍 3：FlashAttention 3 不支持 Blackwell

```
AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90.
Please use `--attention-backend flashinfer`.
```

**解决尝试**：改 `--attention-backend flashinfer` → 引出障碍 4。

### 障碍 4（不可解）：DSpine 硬编码只接受 fa3

```
ValueError: Native DSPINE SWA configuration requires explicit
            --speculative-draft-attention-backend fa3
```

源码 `sglang_dspine/layers/attention.py:6`：
```python
BACKENDS = frozenset(("fa3",))
```

校验函数（同文件 `validate_native_swa_runner`）：
```python
if runner.draft_attention_backend not in BACKENDS:
    raise ValueError("... requires explicit --speculative-draft-attention-backend fa3")
```

**没有绕过路径**——这是源码里的 frozenset 常量，不是配置项。

## 为什么是死锁

```
DSpine 需要 fa3
    ↓
fa3 需要 SM 80-90
    ↓
本机 SM 12.0
    ↓
无解
```

即使改 DSpine 源码把 `flashinfer` 加进 `BACKENDS`，也**没有意义**：DSpine 的 SWA 实现依赖 FA3 的特定 kernel 行为（`window_left=8191`、dense draft KV path），换成 flashinfer 后正确性无人验证，可能产出错误结果。

## 官方文档印证

DSpine 的 Qwen3.8-27B 启动命令写的是：
```
--attention-backend fa3
--speculative-draft-attention-backend fa3
```

而草稿模型 README 的评估环境也写明：
> **Runtime:** SGLang 0.5.22, BF16, **FlashAttention 3**, concurrency 8.

**官方从未验证过 FA3 之外的路径。**

## 可行的替代路径（未验证，仅供参考）

| 方案 | 可行性 | 说明 |
|---|---|---|
| 换 H100/A100 机器 | 高 | 直接满足 SM 80-90 要求 |
| 用 SGLang 内置 MTP 投机解码 | 中 | 模型自带 `mtp_num_hidden_layers: 1`，SGLang 有 `Qwen3_5ForCausalLMMTP`，不走 DSpine 的 fa3 依赖 |
| 等 DSpine 支持 Blackwell | 未知 | 需要上游适配 FA3 或改用 flashinfer |

## 当前系统状态（已恢复）

| 服务 | 端口 | 状态 |
|---|---|---|
| SGLang 0.5.19 + Qwen3.8-27B-NVFP4 | 30000 | ✅ 200（已恢复） |
| SAG2 judge | 8000/8100 | ✅ 200 |
| strata-monitor | 8090 | ✅ 200 |
| stratav4.1（DeepSeek-V4.1） | 8080 | ⏸ 停止（需 `sudo systemctl start stratav4.1`） |

**源码调试补丁已全部撤销**，`kv_cache_configurator.py` 恢复原样（`MEMDBG` 计数为 0）。

## 产物清单

| 路径 | 内容 |
|---|---|
| `/mnt/sda/<user>/dspine_code/DSpine-main/` | DSpine 源码（已 `pip install -e`） |
| `/mnt/sda/<user>/dspine_draft/` | 草稿模型 3.8G（4,051,831,880 B 校验通过） |
| `/mnt/sda/<user>/dspine_wheels/` | 安装包（flashinfer 0.7.1、sglang-kernel 0.4.9 等） |
| `/mnt/nvme1n1/sglang522/` | SGLang 0.5.22 环境（16G） |
| `/mnt/sda/<user>/run_dspine.sh` | 启动脚本（含失败配置，保留供参考） |
| `/mnt/sda/<user>/logs/dspine_*.log` | 各次尝试的完整日志 |

## 一句话总结

**DSpine 的软件栈在本机完全打通（依赖、插件、草稿模型、KV cache 全部就绪），最终卡在 GPU 架构：DSpine 只支持 FlashAttention 3，而 FA3 不支持 Blackwell (sm_120)。这是硬件级的硬约束。**
