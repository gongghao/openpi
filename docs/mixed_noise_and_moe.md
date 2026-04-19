# 混合噪声（Mixed Noise）与 MoE 任务噪声分支

本文档说明 openpi **π0（`Pi0`）** 中可选的 **混合噪声** 机制，以及在混合噪声 **任务分支** 上扩展的 **Mixture-of-Experts（MoE）** 设计。实现位于 `src/openpi/models/pi0.py` 与 `src/openpi/models/pi0_config.py`。

---

## 一、动机

- **标准流匹配**：源分布常取标准高斯 `ε ~ N(0, I)`，与具体任务/机械臂先验无关。
- **混合噪声**：将源噪声拆成 **全局可学习分量**（跨任务共享的「臂/动作」先验）与 **任务相关分量**（由语言 embedding 预测），再用固定系数 `α` 混合，并用 KL 正则把可学习分布拉向标准正态附近。
- **MoE（可选）**：在 **仅任务分支** 上用多个并行 MLP（专家）+ 路由器，按 `task_emb` 做 top‑k 稀疏路由并融合，提高多任务场景下的表达力；**共享分支与 `α` 混合方式不变**。

---

## 二、配置项（`Pi0Config`）

| 字段 | 含义 | 备注 |
|------|------|------|
| `use_mixed_noise` | 是否启用混合噪声 | `False` 时退化为 `ε ~ N(0,I)` |
| `noise_mix_alpha` | 混合系数 `α` | 默认 `0.3` |
| `noise_kl_weight` | KL 项相对 FM 的权重缩放 | 见下文「总损失」 |
| `noise_head_hidden_dim` | 任务分支隐层宽度 | 默认与 action expert 宽度一致 |
| `moe_num_experts` | 任务分支专家数 | **`1` = 单 MLP，不启用 MoE** |
| `moe_top_k` | 每样本激活的专家数 | 仅当 `moe_num_experts > 1` 时生效 |
| `moe_balance_weight` | 负载均衡损失系数 | 仅 MoE 路径 |
| `moe_hidden_dim` | MoE 专家隐层宽度 | 默认同 `noise_head_hidden_dim` 逻辑 |

**启用 MoE 的条件**：`use_mixed_noise=True` 且 `moe_num_experts > 1`。

---

## 三、混合噪声（与 MoE 无关的公共部分）

### 3.1 任务向量 `task_emb`

对 `tokenized_prompt` 做 Gemma **embed**，再按 **mask 做 mean-pool**，得到 `task_emb`，形状 `[B, D]`（`D` 为 PaliGemma 宽度，如 2048）。见 `_get_task_emb`。

### 3.2 共享分支

- 可学习参数：`shared_mu`、`shared_log_sigma`，形状 `[action_horizon, action_dim]`。
- `log_sigma` 在采样与 KL 中 clip 到 `[-2, 2]`。
- 采样：`ε_shared = μ_s + σ_s ⊙ z`，`z ~ N(0,I)`。

### 3.3 任务分支（融合前）

- **单 MLP（`moe_num_experts=1`）**：`task_emb → Linear → swish → mu_head / log_sigma_head`，输出 reshape 为 `[B, H, A]`，`log_sigma` clip 到 `[-2,2]`。
- **MoE（`moe_num_experts>1`）**：见第四节。

### 3.4 混合与采样

设 `α = noise_mix_alpha`：

\[
\varepsilon_{\text{task}} \sim \mathcal{N}(\mu_k,\,\sigma_k^2 I),\quad
\text{noise} = \sqrt{1-\alpha^2}\,\varepsilon_{\text{shared}} + \alpha\,\varepsilon_{\text{task}}.
\]

训练时 `compute_loss` 里对混合噪声使用 `stop_gradient(noise)` 构造流匹配目标速度（与现有实现一致）。

### 3.5 KL 项（单 MLP 模式）

对 **共享** 与 **融合后的任务** 各做对角高斯相对标准正态的 KL（`_diagonal_kl_to_standard`），标量相加得到 `kl_loss`。

---

## 四、MoE 任务分支（`moe_num_experts > 1`）

### 4.1 结构

- **Router**：`noise_router: Linear(D → N)`，`logits = router(task_emb)`，`[B, N]`。
- **专家 `i`**（`i = 0 … N-1`）：与单 MLP 同构的三层结构，参数名为 `noise_expert_hidden_{i}`、`noise_expert_mu_{i}`、`noise_expert_log_sigma_{i}`（使用**具名子模块**而非 `tuple`，以便 `CheckpointWeightLoader` 中 `flatten_dict` 路径全为字符串）。

### 4.2 Top‑k 路由

1. `top_k_values, top_k_indices = top_k(logits, k)`。
2. 仅在 top‑k 的 logits 上做 `softmax`，得到 `k` 个权重（每行和为 1）。
3. 写入长度为 `N` 的 **稀疏权重** `sparse_weights[b, ·]`，其余位置为 0。

### 4.3 专家输出融合

每个专家独立输出 `μ_i`、`log σ_i`（形状 `[B,H,A]`），再按 `sparse_weights` 在专家维上加权求和，得到用于采样的 `μ_k`、`log σ_k`（即与单 MLP 相同接口进入后续混合公式）。

**注意**：实现上会对 **全部 N 个专家** 做前向再按权重融合（非稀疏计算）；top‑k 只影响权重分配。

### 4.4 KL 与负载均衡

- **任务 KL**：对每个专家单独算 `KL_i`，再按 **batch 内 `sparse_weights` 在 batch 上平均后的 `avg_weights[i]`** 加权：  
  `kl_task = sum_i avg_weights[i] * KL_i`。
- **负载均衡（Switch Transformer 风格）**：  
  `L_bal = N * sum_i f_i * p_i`，其中 `f_i` 为 batch 内 **argmax(logits)** 的 top‑1 频率，`p_i` 为 **softmax(logits)** 在 batch 上的平均概率。
- **总 KL**：  
  `kl_loss = kl_shared + kl_task + moe_balance_weight * L_bal`。

---

## 五、训练总损失中的位置

流匹配主损失 `fm_loss` 与 RWFM 等优势加权逻辑不变。若 `use_mixed_noise`：

\[
\text{loss} = \text{fm\_loss} + \frac{\texttt{noise\_kl\_weight} \cdot \texttt{kl\_loss}}{\text{fm\_loss 元素个数}}
\]

即 KL 为 **标量** 均匀摊到每个 FM 元素上（见 `compute_loss`）。

---

## 六、推理

`sample_actions` 在未传入外部 `noise` 时，若启用混合噪声，同样调用 `_compute_mixed_noise`，故 **单 MLP 与 MoE 共用同一入口**，仅任务分支内部实现不同。

---

## 七、Checkpoint 与 `extra_missing_regex`

从 **仅含单任务 MLP**（`task_noise_*`）的旧 checkpoint 加载到 **MoE** 模型时，新参数（`noise_router`、`noise_expert_*`）不存在于 checkpoint，需在 `CheckpointWeightLoader` 中通过 `extra_missing_regex` 声明允许缺失，使这些权重保持初始化值。参见 `training/config.py` 中 `pi0_libero_moe_noise` 等配置示例。

---

## 八、实践建议（简要）

| 场景 | 建议 |
|------|------|
| 极少样本 few-shot SFT | 优先 `moe_num_experts=1`，避免专家与路由欠训 |
| 多任务、数据较充足 | 可尝试 `moe_num_experts=4`、`moe_top_k=2`，并调 `moe_balance_weight` |
| 长程复合技能 | 路由仅依赖 **整条指令** 的 `task_emb`，不随时间步切换；若需阶段化行为，需另设架构或更多数据 |

---

## 九、相关文件

| 文件 | 内容 |
|------|------|
| `src/openpi/models/pi0_config.py` | `Pi0Config` 中混合噪声与 MoE 字段 |
| `src/openpi/models/pi0.py` | `_compute_mixed_noise`、路由、MoE 融合、KL、`compute_loss` |
| `src/openpi/training/config.py` | `pi0_libero_mixed_noise`、`pi0_libero_moe_noise` 等训练配置 |

更多与 LIBERO、RWFM、Classifier Guidance 结合的流程见 `docs/rwfm_classifier_guidance_libero.md`。
