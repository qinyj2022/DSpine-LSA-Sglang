# Qwen3.8-27B 单流吞吐四路分解（dense / +LSA / +DSpine / +LSA+DSpine）

测量日期 2026-11-19，机器 RTX PRO 6000 Blackwell Max-Q（97,887 MiB），sglang 0.5.22 + torch 2.14.1+cu130。

**测量纪律**
- 全部不加 `--decode-log-interval`（逐字日志压 tok/s 10–15%）；每配置独立重启；同一客户端脚本；`temperature=0`。
- 长上下文一律生成 399 token 再计时（只生成 50 个 token 的探针会被首步开销污染，历史 94.55 就是这种产物）。
- 表内每个数字都是同一 prompt 下的对比；**跨上下文行不可比**，因为档案正文不同 → accept 不同。

---

## 1. 短上下文五类负载（bench5.py，tok/s，无日志）

| 配置 | A 对话 | B 摘要 | C 代码 | D JSON | E 数学 | **均值** | 相对 dense |
|---|---|---|---|---|---|---|---|
| 纯 dense（flashinfer，无 LSA、无投机） | 25.17 | 24.64 | 24.73 | 24.51 | 24.57 | **24.72** | 1.00× |
| + LSA（triton，page1，无投机） | 25.30 | 24.79 | 24.53 | 24.21 | 23.95 | **24.56** | **0.99×** |
| + DSpine（flashinfer，无 LSA） | 51.96 | 191.65 | 119.86 | 169.07 | 120.36 | **130.58** | **5.28×** |
| + LSA + DSpine（triton，page1） | 52.42 | 198.18 | 81.01 | 173.41 | 113.17 | **123.64** | **5.00×** |

## 2. 长上下文 tok/s（decode_probe.py `400 long`，399 token 解码，prefill 命中 radix 缓存）

| 上下文 | dense | +LSA | +DSpine（flashinfer） | 对照：triton+page1 不开 LSA | +LSA+DSpine |
|---|---|---|---|---|---|
| 16k | 24.42 | 24.27 | 70.87 / 74.86 | 49.81 | 66.83 / 69.63 |
| 64k | 23.04 | 23.46 | 87.50 / 88.47 | 28.68 / 28.77 | 79.17 / 82.10 |
| 118k | 21.83 | 23.29 | 72.68 | **15.90** | 66.39 / 73.75 |

## 3. 每步耗时归因（ms/step，batch=1，log 归一化，与 accept 无关）

`step_ms = 1000 × accept_len / gen_throughput`，从每步日志直接算，是被测配置自身可归因的成本。

| 上下文 | DSpine + flashinfer（无 LSA） | triton + page1（无 LSA） | **LSA + DSpine** | LSA 相对 flashinfer |
|---|---|---|---|---|
| 16k | 未测 | 80.6 | **52.2** | — |
| 64k | 55.6 | 180.5 | **53.8** | −3.2% |
| 118k | 59.6 | **296.7** | **54.3** | **−8.9%** |

LSA+DSpine 的步耗时从 16k 到 118k **完全平坦（52→54 ms）**；flashinfer 路线则随上下文爬升（55.6→59.6），
triton+page1 无索引则爆炸（80.6→296.7，5.5×）。

## 4. 结论（三条，均以步耗时为准）

1. **+LSA 单独（无投机）**：短上下文 0%（24.56 vs 24.72）；118k +6.7%（23.29 vs 21.83）。单流纯解码下 LSA 收益很小。
2. **+DSpine 单独**：24.72 → 130.58（**5.28×**）；118k 从 21.83 → 72.68（3.3×）。**单流 tok/s 的收益几乎全部来自投机解码。**
3. **LSA 的真实定位**：它强制 `triton + page-size 1`，而这个后端在没有索引时随上下文线性恶化
   （118k 296.7 ms/步 → 15.9 tok/s）。LSA 把同等配置拉回 54.3 ms/步（**5.5× 修复**），
   并且比"不开 LSA 的最优配置（flashinfer）"还快 **8.9%**（118k）、3.2%（64k）。

## 5. 对历史结论的更正

| 历史说法 | 更正 |
|---|---|
| "LSA 在 119k 给 DSpine 带来 1.29–1.46×" | **不成立**。那组 tok/s 来自 50 余 token 的短探针，首步开销污染分母。按步耗时 LSA 的真实收益是 **8.9%/步（118k）**，换算 tok/s 在 accept 不变时约 +9% |
| "graph_lsa.sh @119k = 94.55 tok/s（n=5 中位）" | **不可复现**。同一脚本今天 3 次读数 74.67 / 74.67 / 60.22。步耗时仍是 54.3 ms（与历史 54 ms 一致），所以差异全在 accept，不在配置性能 |
| "16k 比 120k 快" | 假象，纯 accept 差异 |

## 6. 选择建议

| 目标 | 配置 |
|---|---|
| 单流最快、上下文 ≤64k | `bf16_src_uds.sh`（flashinfer + DSpine，无 LSA）：64k 下步耗时只差 3%，tok/s 常更高 |
| 上下文 ≥100k | `graph_lsa.sh`：步耗时平坦且低 8.9%，accept 相同的条件下更快 |
| 必须 page_size 1 / triton（FP4 KV、自定义 mask） | **必须开 LSA**，否则 118k 只有 15.9 tok/s |
| 高并发吞吐 | 未做隔离对照（combo 历史记录 606 tok/s @16 并发），是目前唯一可能翻盘的口子 |

## 7. 复现命令

```bash
# 五类负载（短上下文）
python tools/bench5.py
# 长上下文（399 token，稳定样本；末尾 long 用长输出问题）
python tools/decode_probe.py 118000 400 long
# 每步耗时归因（需 --decode-log-interval 1 启动）
python step_stats.py logs/<boot>.log 8
```

脚本：`tools/ctrl_ds_triton_p1.sh`（隔离对照，DSpine + triton + page1 但**不**开 LSA）、
`tools/step_stats.py`（步耗时归因）。服务器端副本：`/mnt/sda/<user>/ctrl_ds_triton_p1.sh`、`step_stats.py`。
