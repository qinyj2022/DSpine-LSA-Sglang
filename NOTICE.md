# NOTICE

本仓库包含或派生自以下第三方开源项目。原始著作权归各自作者所有，均以 **Apache License 2.0** 授权。

## 1. DSpine

- 来源：https://github.com/tencent/dspine
- 论文：*Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding*，arXiv:2609.36173
- 著作权：Copyright (C) 2026 Tencent
- 许可：Apache License 2.0（原文见 `src/dspine_code/DSpine-main/License.txt`）
- 位置：`src/dspine_code/DSpine-main/`（插件源码，基本为上游原样）
- 本仓库的修改：`sglang_dspine/plugin.py` 中标注 `[PATCH]` 与 `[PATCH dspine-stable]` 的行
  （草稿图在 FlashInfer SWA 路径下强制 eager；采样轮缺草稿链时回落到原生 DFlash verify）

## 2. latent-sparse-attention (LSA)

- 来源：https://github.com/Prefrontal-AI/latent-sparse-attention
- 许可：Apache License 2.0（原文见 `patches/lsa/LICENSE`）
- 位置：`patches/lsa/`（仅包含带 `[PATCH spec ...]` 标记的改动文件，不含上游完整仓库）
- 且该上游包含来自 flash-linear-attention 的 MIT 许可代码，其第三方声明随附于
  `patches/lsa/THIRD_PARTY_NOTICES.md`
- 本仓库的修改：14 个文件中标注 `[PATCH spec ...]` 的行（详见 README 第 9 节）

## 3. SGLang

- 来源：https://github.com/sgl-project/sglang
- 许可：Apache License 2.0
- 位置：`patches/sglang/srt/speculative/eagle_worker_common.py`
  （SGLang 侧唯一改动文件，改动行标注 `[PATCH spl]` / `[PATCH spl2]` / `[PATCH spl3]`）
- SGLang 其余部分未随本仓库分发

## 4. 模型权重

不随本仓库分发。使用需自行获取：

- 目标模型：`Qwen/Qwen3.8-27B`
- DSpine 草稿：`tencent/Qwen3.8-27B-DSpine-b16`

各自的许可条款以 Hugging Face 模型卡为准。

## 5. 本仓库自身内容

补丁脚本（`src/lsa_code/patch_*.py`）、启动脚本（`scripts/*.sh`）、测量工具（`tools/`）
与文档（`docs/`、`README.md`）由本仓库作者编写，随仓库以 Apache License 2.0 提供。
