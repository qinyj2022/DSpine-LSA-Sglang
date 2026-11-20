# Qwen3.8-27B 在 SGLang 上的部署报告

日期：2026-11-13
主机：`<gpu-host>`（`<gpu-host>`，Ubuntu 22.04，kernel 6.8.0-124）
GPU：NVIDIA RTX PRO 6000 Blackwell Max-Q，97,887 MiB（cc 12.0 / sm_120），驱动 595.99.02
结论：**成功**。Qwen3.8-27B（NVFP4）已在 SGLang 0.5.19 上运行于 `0.0.0.0:30000`，OpenAI 兼容 API，真实推理通过。

## 1. 起点：三个被澄清的前提

任务开始时存在三个与实机不符的前提，实测后逐一纠正：

| 原说法 | 实测结果 |
|---|---|
| "系统部署了 vLLM" | vLLM 只存在于 `dsv4-vllm-1layer`（为 DeepSeek-V4 建的），**与 Qwen3.8 无关** |
| "DSpine / BSPine 草稿系统" | 三个 Qwen3.8 模型目录中 grep **零命中**；官方是 **MTP**（`mtp_num_hidden_layers: 1`，`model_mtp.safetensors`） |
| "LSA 稀疏注意力" | 搜索 `LSA` / `latent_sparse` / `sparse_attention` **零命中**，不存在 |

模型位于 `/mnt/sda/models/`（不是 `/mnt/nvme1n1/`），三个变体：

| 变体 | 大小 | 格式 |
|---|---|---|
| `Qwen3.8-27B` | 52 GB | BF16 原始，18 分片 |
| `Qwen3.8-27B-FP8` | 22 GB | mixed-precision（FP8 + NVFP4） |
| `Qwen3.8-27B-NVFP4` | 22 GB | 同上，**本次采用** |

## 2. 模型真实架构（实测 config.json）

```
architectures         : Qwen3_5ForConditionalGeneration
model_type            : qwen3_5 / qwen3_5_text
num_hidden_layers     : 64
layer_types           : 48 × linear_attention + 16 × full_attention
                        (full_attention_interval = 4 → Gated DeltaNet → Gated Attention 循环)
hidden_size           : 5120        num_attention_heads : 24    num_key_value_heads : 4
head_dim              : 256         max_position_embeddings : 262144
vocab_size            : 248320      mtp_num_hidden_layers : 1
vision                : qwen3_5_vision, depth 27, patch 16
quant_method          : compressed-tensors, format: mixed-precision
```

注意：README 自称 "dense model"，但实际是 **Gated DeltaNet（线性注意力）与 Gated Attention 混合架构**，不是纯稠密 Transformer。

## 3. 环境构建：复用现有 conda env

**关键决策**：不新建 venv，而是克隆现有 `dsv4-vllm-1layer`（8.9 GB）。

理由：SGLang 0.5.19 精确要求 `torch==2.13.0`，而该环境已有 `torch 2.13.0+cu130`——最容易出问题的环节直接省掉。

```bash
# 克隆（原 env 的 python3.12 是内嵌的完整二进制，权限 777）
cp -a /mnt/nvme1n1/conda-envs/dsv4-vllm-1layer/. /mnt/nvme1n1/sglang312/

# 验证隔离正确
/mnt/nvme1n1/sglang312/bin/python -c "import sys;print(sys.prefix)"   # → /mnt/nvme1n1/sglang312
```

**必须用 `python -m pip` 调用**：`bin/pip` 的 shebang 仍指向旧环境路径，直接执行会污染源环境。

## 4. 安装过程踩到的坑（按顺序）

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 1 | 清华源下载 377MB 的 `sglang-kernel` 卡死 8 分钟 | 大文件限速（~35 KB/s） | 改用阿里云镜像，11 MB/s |
| 2 | `nvidia_cublas` 423MB 下到 303MB 断连 | 网络抖动 | 用 `--resume-retries` 重试 |
| 3 | `total archs: 0`，257 个模型全导入失败 | 缺 `gguf` 模块 | `pip install gguf` |
| 4 | 同上，另一个原因 | `transformers 5.17.0` 不兼容（SGLang 要 5.12.1） | 降级 `transformers==5.12.1` |
| 5 | 同上，第三个原因 | 缺 `sgl_kernel`（`sglang-kernel` 包） | 安装本地已验证的 wheel |
| 6 | `AutoProcessor` 导入失败 | `librosa` 缺 `lazy_loader`（之前 `--no-deps` 装的） | 补装 `lazy_loader` |
| 7 | `FileNotFoundError: 'ninja'` | venv bin 不在 PATH | `export PATH=/mnt/nvme1n1/sglang312/bin:$PATH` |
| 8 | `nvcc: fatal error: writing output to /tmp/...: Invalid argument` | **根分区 `/` 100% 满**（466/492G） | 清理 pip 缓存，把 TMPDIR 与所有编译缓存指向 `/mnt/nvme1n1` |

第 8 条是最隐蔽的：ninja 看起来"失败"，实际是磁盘满导致 nvcc 无法写临时文件。

## 5. 最终启动脚本

保存于 `/mnt/nvme1n1/run_qwen38_sglang.sh`：

```bash
#!/bin/bash
export PATH=/mnt/nvme1n1/sglang312/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=8
# 缓存全部放 880G 的大盘，不放 99% 满的根分区
export TMPDIR=/mnt/nvme1n1/sglang_cache/tmp
export SGLANG_CACHE_DIR=/mnt/nvme1n1/sglang_cache/sglang
export FLASHINFER_CACHE_DIR=/mnt/nvme1n1/sglang_cache/flashinfer
export TRITON_CACHE_DIR=/mnt/nvme1n1/sglang_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/nvme1n1/sglang_cache/inductor
export XDG_CACHE_HOME=/mnt/nvme1n1/sglang_cache/xdg
mkdir -p $TMPDIR $SGLANG_CACHE_DIR $FLASHINFER_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME

MODEL=/mnt/sda/models/Qwen3.8-27B-NVFP4
exec /mnt/nvme1n1/sglang312/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --trust-remote-code \
  --served-model-name qwen3.8-27b
```

启动命令：
```bash
setsid nohup /mnt/nvme1n1/run_qwen38_sglang.sh > /mnt/nvme1n1/qwen38_run.log 2>&1 < /dev/null &
```

## 6. 运行实测

**加载指标**
```
Load weight end. elapsed=6.98 s (缓存命中；首次 131.29 s), type=Qwen3_5ForConditionalGeneration,
                 quant=compressed-tensors, avail mem=68.98 GB, mem usage=21.48 GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 957112, K size: 14.60 GB, V size: 14.60 GB
GPU 驻留: 88,089 MiB / 97,887
```

**`GET /v1/models`**
```json
{"object":"list","data":[{"id":"qwen3.8-27b","object":"model","owned_by":"sglang",
 "root":"qwen3.8-27b","parent":null,"max_model_len":32768}]}
```

**`POST /v1/chat/completions`（真实推理）**
```
prompt: "What is the capital of France? Answer in one short sentence."
→ "The capital of France is Paris."   (41 completion tokens, finish_reason=stop)
```

**吞吐**
```
200 token 生成：162 completion tokens / 4.06 s ≈ 40 tok/s
```

## 7. 关键结论：SGLang 0.5.19 的架构支持（实测）

```
TOTAL_ARCHS: 257
QWEN_ARCHS: 28
  Qwen3_5ForConditionalGeneration   ← 本次使用的架构
  Qwen3_5ForCausalLMMTP             ← MTP 投机解码
  Qwen3_5MoeForConditionalGeneration
  Qwen3_5ForCausalLM
  （另有 Qwen3Next / Qwen3VL / Qwen3Moe 等）
```

**SGLang 0.5.19 原生支持 Qwen3.8-27B 的架构和 MTP。** 这与模型 README 的声明一致：
> "This model works with vLLM. It also works with SGLang, but only with the v0.5.19, because the `lm_head` is FP8 rather than 4-bit."

版本锁定是真实的，实测证明 0.5.19 可用。

## 8. 副作用与回滚

**已停止的服务**：`stratav4.1`（DeepSeek-V4.1-Flash，原占 91.9 GB 显存，`enabled` 状态）。
服务本身未被修改，仅停止。恢复：
```bash
sudo systemctl start stratav4.1     # 需 sudo 密码
```

**磁盘**：清理了 `~/.cache/pip`（3.7 GB）。根分区从 100% 降到 99%（6.5 GB 可用），仍很紧张。

**未修改**：`dsv4-vllm-1layer` 原环境完全未动（只读复制）；其他用户（wyq）环境未触碰。

**彻底回滚**（如需释放 8.9 GB + 300MB）：
```bash
rm -rf /mnt/nvme1n1/sglang312 /mnt/nvme1n1/sglang_cache /mnt/nvme1n1/whls
```

## 9. 遗留风险

1. **根分区 99% 满**（6.5 GB 可用）。这是部署期间最危险的约束，任何大文件写入都可能失败。建议清理。
2. **服务未做 systemd 常驻**：当前是 `setsid nohup` 独立进程，**重启机器后不会自动起来**。
3. **显存 88 GB / 97.9 GB**，与 `stratav4.1` **不能同时运行**。要跑 DeepSeek-V4.1 需先停 SGLang。
4. **MTP 未启用**：本次启动未加 `--speculative-algorithm`，MTP 层存在但未用于投机解码。加 MTP 可望提速，需另测。
5. **上下文设为 32768**（模型原生支持 262,144）。增大需重新评估显存。
6. **`transformers` 被降到 5.12.1**（仅影响 sglang312 副本，原 vLLM 环境仍是 5.17.0）。

## 10. 产物清单

| 路径 | 内容 |
|---|---|
| `/mnt/nvme1n1/sglang312/` | SGLang 0.5.19 运行环境（8.9 GB，克隆自 dsv4-vllm-1layer） |
| `/mnt/nvme1n1/run_qwen38_sglang.sh` | 启动脚本 |
| `/mnt/nvme1n1/sglang_cache/` | 编译缓存（TMPDIR / flashinfer / triton / inductor） |
| `/mnt/nvme1n1/qwen38_run6.log` | 本次成功启动的日志 |
| `/mnt/nvme1n1/whls/` | 已验证的 `sglang_kernel` wheel（SHA256 校验通过） |
