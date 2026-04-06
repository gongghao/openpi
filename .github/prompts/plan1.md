---

### 需要新增/修改的文件清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/openpi/models/value_function.py` | **新建** | 分布式 VF + Action-Conditioned VF |
| `src/openpi/models/pi0.py` → `compute_loss` | **修改** | 添加 `advantages` 参数和加权逻辑 |
| `src/openpi/models/pi0.py` → `sample_actions` | **修改** | 添加 classifier guidance 逻辑 |
| `src/openpi/models/model.py` → `BaseModel` | **修改** | 更新 `compute_loss` 签名 |
| `scripts/train.py` → `train_step` | **修改** | 传递 advantages 给 loss |
| `src/openpi/training/config.py` | **修改** | 添加 RWFM / CG 配置字段 |
| `src/openpi/training/data_loader.py` | **修改** | 支持加载 advantage 字段 |
| `scripts/collect_libero_data.py` | **新建** | rollout + 奖励标注 + 存储 |
| `scripts/train_value_function.py` | **新建** | VF 训练入口 |
| `scripts/precompute_advantages.py` | **新建** | 预计算 advantage 值并写入数据集 |

---

### 论文叙事结构建议

1. **Motivation**：RECAP 用二值优势条件化，丢失了连续优势信息；πRL 的 ELBO 近似引入了额外误差和 trust region 约束。我们提出一种更直接的方式。
2. **Method**：
   - 训练时：**Reward-Weighted Flow Matching** 用连续优势值直接加权去噪损失，保留全部数据（不丢弃任何 trajectory），且引入噪声自适应权重
   - 推理时：**Classifier Guidance** 用价值函数梯度引导 ODE 采样，两端互补
3. **Theory**：证明 RWFM 等价于一种隐式的正则化 RL 目标（类似 AWR 但避免了加权回归对数据的浪费）
4. **Experiments**：LIBERO 上的全面消融实验，证明训练端和推理端的改进是互补的

需要我帮你**开始实际编写代码到文件中**，还是先对某个 Phase 做更深入的讨论？