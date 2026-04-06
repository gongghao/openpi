# LIBERO 离线数据上的高产出 RL 方案：奖励加权 Flow Matching + Classifier Guidance

本文档面向在 **openpi（π0）** 代码库上、使用 **LIBERO 仿真环境离线数据集** 训练的场景，给出「奖励加权 Flow Matching（RWFM）」与「Classifier Guidance（CG）」组合的**可落地实现方案**。

---

## 一、前置条件与目标

| 项目 | 说明 |
|------|------|
| 代码库 | `openpi-main`，JAX 主训练路径，`Pi0.compute_loss` / `sample_actions` |
| 数据 | LeRobot 格式 LIBERO 数据（如 `physical-intelligence/libero`），`prompt_from_task=True` |
| 基线 | 纯模仿学习（BC）流匹配训练，无 RL |
| 目标 | 在**不依赖** πRL 式 ELBO+PPO 的前提下，用**训练端加权** + **推理端价值梯度引导**提升策略 |

---

## 二、方案总览

### 2.1 核心思路

1. **Reward-Weighted Flow Matching（RWFM，训练）**  
   在标准 flow matching 损失上乘以由优势（advantage）导出的权重，使「高回报轨迹上的动作」在去噪目标中获得更大梯度；可选**噪声水平自适应**权重（低噪声阶段更强调优势）。

2. **Classifier Guidance（推理）**  
   使用（或单独训练）**动作条件**价值网络 \(V_\phi(s, x_t, t)\)，在 ODE 采样每一步将 \(-\lambda(t)\nabla_{x_t} V\) 叠加到向量场 \(v_\theta\) 上，使采样偏向高价值动作区域。

### 2.2 与 RECAP / πRL 的关系

- **RECAP（π\*0.6）**：二值优势条件化 + 监督式策略提取，不直接对 FM loss 做连续加权。  
- **πRL**：对数似然下界 + 策略梯度。  
- **本方案**：**连续优势加权 FM**（训练）+ **价值梯度引导采样**（推理），实现简单、与现有 `compute_loss` / `sample_actions` 改动面集中。

---

## 三、Phase 0：数据采集与奖励标注

### 3.1 用途

若仅用**专家演示**数据，可令每条轨迹 `success=True`、回报为「到成功剩余步数」的负值；若后续有 **rollout** 数据，则需区分成功/失败 episode。

### 3.2 轨迹级数据结构（建议）

```python
{
    "observations": [...],       # T 步：image, wrist_image, state
    "actions": [...],            # 与 openpi 一致的 action chunk 或逐步动作
    "rewards": [...],            # 每步奖励
    "returns": [...],            # Monte Carlo return G_t
    "success": bool,
    "task_description": str,
    "episode_length": int,
}
```

### 3.3 奖励定义（与 RECAP 式稀疏奖励一致）

- \(r_T = 0\)（成功）或 \(-C_{\text{fail}}\)（失败）  
- \(r_t = -1\)（\(t < T\)）

```python
def compute_rewards(episode_steps, success, c_fail=200):
    T = len(episode_steps)
    rewards = np.full(T, -1.0)
    if success:
        rewards[-1] = 0.0
    else:
        rewards[-1] = -c_fail
    return rewards

def compute_returns(rewards):
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G += rewards[t]
        returns[t] = G
    return returns
```

### 3.4 归一化

将回报归一化到约 \((-1, 0)\)（按任务最大步长缩放），便于与分布式价值函数的 bin 对齐：

```python
def normalize_returns(returns, max_episode_length):
    return returns / max_episode_length
```

---

## 四、Phase 1：价值函数（Value Function）

### 4.1 作用

- **Phase 2**：由 \(V(s)\) 与回报估计优势 \(A \approx G_t - V(s_t)\)，用于 RWFM 权重。  
- **Phase 3（可选）**：若只做 state 价值，推理端需 **action-conditioned** 价值网络才能对 \(x_t\) 求梯度。

### 4.2 分布式价值（RECAP 风格，简化版）

- 输出 \(p_\phi(V \mid o, \ell)\) 在 \(B\) 个 bin 上的类别分布；期望价值 \(\hat{V} = \sum_b p_b \, v(b)\)。  
- 损失：将目标回报离散到 bin 上，做交叉熵。

### 4.3 实现建议路径

- **轻量路径**：独立小网络（SigLIP 冻结特征 + state + 语言嵌入池化）+ MLP head。  
- **Action-conditioned VF（用于 CG）**：额外输入 `noisy_actions` 展平向量 + flow 时间 \(t\)，输出标量 \(V(s, x_t, t)\)。

### 4.4 Advantage

- **MC**：\(A_t = G_t^{\text{norm}} - V(s_t)\)。  
- **N-step**（可选）：\(A_t = \sum_{t'=t}^{t+N-1} r_{t'} + V(s_{t+N}) - V(s_t)\)（需轨迹对齐）。

---

## 五、Phase 2：奖励加权 Flow Matching（RWFM）

### 5.1 损失形式

标准 FM（每步）：

\[
\mathcal{L}_{\text{FM}} = \|v_\theta(x_t, t, s) - u_t\|^2
\]

加权：

\[
\mathcal{L}_{\text{RWFM}} = w(A) \cdot \mathcal{L}_{\text{FM}}
\]

权重示例（自归一化 exp 优势）：

\[
w(A) = \frac{\exp(A/\beta)}{\mathbb{E}_B[\exp(A/\beta)]}
\]

### 5.2 噪声自适应权重（可选）

令 \(\eta\) 为 flow 时间（与代码中 `time` 一致），例如：

\[
w'(A, \eta) = 1 + (1-\eta)\,(w(A) - 1)
\]

直觉：接近干净动作（\(\eta\) 小）时更强调优势差异。

### 5.3 与 `openpi` 的对接点

在 `src/openpi/models/pi0.py` 的 `compute_loss` 中，在得到 `fm_loss`（形状 `[B, action_horizon]`）之后，乘以 batch 级权重 `advantages`（需从数据管道传入）。

`compute_loss` 建议新增可选参数：

- `advantages: [B] | None`  
- `rwfm_beta: float`  
- `rwfm_noise_adaptive: bool`

### 5.4 数据管道

- 在样本中增加字段 `advantages`（或预计算后写入扩展数据集 / 旁路 `.npz` 索引）。  
- 修改 `scripts/train.py` 的 `train_step`：`batch` 中解出 `advantages` 并传入 `model.compute_loss`。  
- 修改 `src/openpi/training/data_loader.py` 的 batch 组装逻辑，使 `advantages` 与 `Observation`、`actions` 对齐。

---

## 六、Phase 3：Classifier Guidance（推理）

### 6.1 修改后的 ODE 步（概念）

原：`x \leftarrow x + dt \cdot v_\theta(x, t)`  

引导：

\[
x \leftarrow x + dt \cdot \bigl(v_\theta(x,t) - \lambda(t)\,\nabla_x V_\phi(s, x, t)\bigr)
\]

（具体符号与 `dt` 正负需与 `Pi0.sample_actions` 中「\(t=1\) 噪声 → \(t=0\) 动作」约定一致。）

### 6.2 引导强度

例如 \(\lambda(t) = \lambda_0 (1-t)\)：高噪声时弱引导，接近解时强引导。

### 6.3 与 `openpi` 的对接点

在 `src/openpi/models/pi0.py` 的 `sample_actions` 内 `step` 函数中，在得到 `v_t` 后：

1. 用 `jax.grad` 对 `x_t` 求 \(V_\phi(s, x_t, t)\) 的梯度（需 action-conditioned VF）。  
2. `v_t = v_t - lambda_t * grad_V`（符号以实现约定为准）。  

`sample_actions` 建议新增可选参数：

- `value_fn`（可调用模块）  
- `guidance_scale`（\(\lambda_0\)）  
- `guidance_noise_adaptive`

### 6.4 策略封装

在 `openpi.policies.policy.Policy` 的 `infer` / `sample_kwargs` 中传入 `guidance_scale` 等，便于 `serve_policy` 与 `examples/libero` 评估脚本统一配置。

---

## 七、Phase 4：训练配置与实验

### 7.1 `TrainConfig` 建议字段（`config.py`）

| 字段 | 含义 |
|------|------|
| `rwfm_enabled` | 是否启用 RWFM |
| `rwfm_beta` | 优势温度 \(\beta\) |
| `rwfm_noise_adaptive` | 是否噪声自适应权重 |
| `guidance_scale` | 推理引导强度（若在服务里统一读配置） |
| `value_fn_checkpoint` | 价值网络检查点路径 |

### 7.2 训练流程（建议顺序）

1. **Phase 0**：准备带 `returns` / `success` 的轨迹或标注表。  
2. **Phase 1**：训练 state（或 state+lang）价值函数；预计算每条样本的 `advantage`。  
3. **Phase 2**：在 BC 初始化权重上继续训练 π0，启用 RWFM。  
4. **Phase 1b（若用 CG）**：训练 action-conditioned \(V(s,x_t,t)\)。  
5. **评估**：LIBERO `main.py` / `main_local.py` 上对比 BC vs RWFM vs RWFM+CG。

### 7.3 消融实验表

| ID | 训练 | 推理 | 目的 |
|----|------|------|------|
| E1 | BC | 标准 | 基线 |
| E2 | RWFM | 标准 | 验证训练端加权 |
| E3 | BC | CG | 验证仅推理引导 |
| E4 | RWFM | CG | 完整组合 |
| E5 | RWFM（非噪声自适应） | 标准 | 消融权重调度 |
| E6 | RWFM | CG（固定 \(\lambda\)） | 消融 \(\lambda(t)\) |

### 7.4 超参搜索建议

- `rwfm_beta`：\(\{0.1, 0.5, 1.0, 2.0, 5.0\}\)  
- `guidance_scale`：\(\{0, 0.01, 0.05, 0.1, 0.5\}\)  
- VF `hidden_dim`：\(\{128, 256, 512\}\)  
- 分布式 bin 数：\(\{51, 101, 201\}\)

---

## 八、建议新增 / 修改的文件清单

| 路径 | 操作 | 说明 |
|------|------|------|
| `src/openpi/models/value_function.py` | 新建 | 分布式 VF、可选 action-conditioned VF |
| `src/openpi/models/pi0.py` | 修改 | `compute_loss` 加权；`sample_actions` 引导 |
| `src/openpi/models/model.py` | 修改 | `BaseModel.compute_loss` 抽象签名与默认参数 |
| `scripts/train.py` | 修改 | `train_step` 传入 `advantages` |
| `src/openpi/training/config.py` | 修改 | RWFM / VF / guidance 相关配置 |
| `src/openpi/training/data_loader.py` | 修改 | batch 含 `advantages` |
| `scripts/collect_libero_data.py` | 新建（可选） | rollout + 标注 |
| `scripts/train_value_function.py` | 新建（可选） | VF 训练入口 |
| `scripts/precompute_advantages.py` | 新建（可选） | 离线预计算优势并写回数据 |

---

## 九、论文叙事要点（可选）

1. **动机**：RECAP 用二值条件化；πRL 依赖似然下界与 PPO 类约束。本方案在 FM 上直接做**连续优势加权**，推理端用**价值梯度**对齐高回报区域，工程改动集中。  
2. **贡献**：噪声自适应 RWFM；与 classifier guidance 在 flow 采样上的组合；LIBERO 上系统消融。  
3. **局限**：依赖价值估计质量；action-conditioned VF 需额外数据与训练；引导过强可能导致动作越界，需调 \(\lambda\) 与裁剪。

---

## 十、代码参考锚点（当前仓库）

- Flow matching 训练：`src/openpi/models/pi0.py` → `compute_loss`（`fm_loss = mean(square(v_t - u_t))`）。  
- 采样 ODE：`src/openpi/models/pi0.py` → `sample_actions`（`while_loop` + `step` 内 `v_t`）。  
- 训练步：`scripts/train.py` → `train_step`。  
- 数据 batch：`src/openpi/training/data_loader.py` → `DataLoaderImpl.__iter__`。  
- LIBERO 数据与变换：`src/openpi/training/config.py` → `LeRobotLiberoDataConfig`；`src/openpi/policies/libero_policy.py`。

---

*文档版本：与对话中「高产出组合」方案一致，便于后续按 Phase 逐步实现。*
