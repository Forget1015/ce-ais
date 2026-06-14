# CE-AIS 性能提升系统评估与改进计划

## 1. 当前状态总结

### 1.1 已完成的基础设施修复

| 问题 | 修复内容 | 影响 |
|------|----------|------|
| calvin_env 版本落后 | 从 `1431a46` 升级到 FLOWER 指定的 `797142c` | 修复了 button/light reset 顺序 + action in-place 修改 bug |
| 图像预处理不一致 | 对齐为 `Resize(224) → /255 → Normalize`，去除 RandomShiftsAug | 结果确定性，可复现 |
| 权重加载 key mismatch | 处理了 `vlm.language_final_logits_bias` 等 key 映射 | 模型加载无 warning |

### 1.2 Frozen FLOWER Baseline 数字（seed=42, 1000 chains, calvin_env 797142c）

来源：`results/flower_official_1000chains/merged_results.json`

| L1 | L2 | L3 | L4 | L5 | Avg Len |
|----|----|----|----|----|---------|
| 97.5% | 91.8% | 85.5% | 79.7% | 73.6% | 4.281 |

注：该结果使用旧版 calvin_env 跑的。新版 calvin_env 下的数字正在 multi_seed_eval 中验证。

### 1.3 CE-AIS 当前表现

**保守参数（base.yaml 默认）**：

| L1 | L2 | L3 | L4 | L5 | Avg Len | accepted_rate |
|----|----|----|----|----|---------|---------------|
| 97.7% | 92.3% | 86.1% | 80.2% | 73.3% | 4.296 | 99.75% |

结论：几乎全部 pass-through，提升 0.015，无统计意义。

**激进参数（aggressive_steering.yaml）**——50 chains 快速测试：

| L1 | L2 | L3 | L4 | L5 | Avg Len | accepted_rate |
|----|----|----|----|----|---------|---------------|
| 98.0% | 94.0% | 88.0% | 82.0% | 70.0% | 4.32 | 100% |

对比同 50 chains 的 frozen_flower baseline（seed=42, 新 calvin_env）：

| L1 | L2 | L3 | L4 | L5 | Avg Len |
|----|----|----|----|----|---------|
| 98.0% | 94.0% | 88.0% | 82.0% | 76.0% | 4.38 |

结论：L1-L4 持平，L5 下降 6%。Steering 力度放开后 energy 确实降了（-1.03 → -1.18），但 **energy 降低 ≠ 任务成功**。

### 1.4 核心问题诊断

**CE-WM 的 energy landscape 和真实任务成功率不对齐。**

原因：
- 训练时正样本 = expert 轨迹，负样本 = 随机扰动的 action
- CE-WM 学到的是"区分 expert 和随机噪声"
- FLOWER 的 action 本身已经接近 expert 分布，CE-WM 在 VLA action 附近的能量梯度无意义
- Steering 沿着"远离随机噪声"的方向走，但这个方向和"任务成功"无关，甚至有害

## 2. 改进方案

### 2.1 方案一：基于 Rollout 成功/失败信号重训 Energy Head（推荐首选）

**核心思路**：让 CE-WM 的 energy 直接反映"action 在当前 state 下是否导致任务成功"。

**具体步骤**：

1. **收集 rollout 数据**
   - 用 frozen_flower + seed=42 跑 1000 chains × 5 tasks
   - 记录每步的 (z_t, a_t, task_success) 三元组
   - z_t 由冻结的 Encoder 计算
   - a_t 是 FLOWER 实际输出的 action
   - task_success = 该 step 所在的 subtask 最终是否成功

2. **构造正负样本**
   - 正样本：成功 subtask 中所有 step 的 (z_t, a_t)
   - 负样本：失败 subtask 中所有 step 的 (z_t, a_t)
   - 特别是：失败 subtask 的**后期步骤**（接近 timeout 的）作为强负样本

3. **微调策略**
   - 只训练 Energy Head MLP（`src/world_model/energy_head.py`）
   - 冻结 Mamba-3 backbone（保留其时序建模能力）
   - Loss 仍用 calibrated NCE，但正负样本来源改变
   - 训练量小（MLP 只有 ~200K 参数），几分钟内完成

4. **评估**
   - 重新跑 CE-AIS eval，对比 steering 后的 avg_len
   - 关注 energy_before/after 是否和 task success 相关

**优势**：
- 不改架构，不改创新点
- 让 energy 有了真实的任务语义
- Steering 方向从"远离随机噪声"变成"远离失败模式"
- 训练成本极低

**风险**：
- 失败样本数量有限（1000 chains 中约 264 个 non-5 chains）
- 可能过拟合到特定的失败模式

### 2.2 方案二：改进负样本构造策略

**核心思路**：不依赖 rollout 数据，而是在训练时构造更有意义的负样本。

**具体做法**：
- 当前负样本：对 expert action 加高斯噪声（`neg_sample_ratio=5`）
- 改为：
  - 方向性扰动：沿着常见失败方向（如 push/rotate 任务中的反方向）
  - 时序不一致：打乱 action sequence 的时间顺序
  - 幅度衰减：让负样本和正样本的差距更小（当前可能太大，导致 decision boundary 远离 VLA action 区域）

**优势**：不需要 rollout 数据，可以直接重训

**风险**：需要人工设计"失败方向"，泛化性不确定

### 2.3 方案三：多步 Energy 评估

**核心思路**：单步 E(z_t, a_t) 无法判断长期后果，改为 E(z_{t-N:t}, a_{t-N:t})。

**具体做法**：
- 给 CE-WM 输入最近 N 步（如 N=10）的 (z, a) 序列
- Energy 评估的是"整段轨迹是否合理"
- Steering 时只修改最后一步的 action，但评估考虑了历史上下文

**优势**：Mamba-3 本身就是序列模型，天然适合多步评估

**风险**：计算量增加，需要重训

### 2.4 方案四：Steering 参数自适应

**核心思路**：不同任务/场景用不同的 steering 强度。

**观察**：失败主要集中在 push/rotate block 任务（需要精细力控），而 lift/open_drawer 等任务基本不失败。

**做法**：
- 根据 uncertainty 动态调整 steering 参数
- 高 uncertainty → 更大的 step_size 和 n_steps（更积极介入）
- 低 uncertainty → 更小的 step_size（基本不动）
- 这和 v1 论文中的 bilateral gating 一致，只是需要更好地校准阈值

## 3. 实验计划

### Phase 1：数据收集（当前进行中）

- [x] 修复 calvin_env 到 797142c
- [x] 去除 RandomShiftsAug 确保确定性
- [ ] multi_seed_eval 跑完（验证 frozen_flower 在不同 seed 下的分布）
- [ ] 用 seed=42 跑一次完整的 frozen_flower 1000 chains × L5，并保存每步的 (z_t, a_t)

### Phase 2：Energy Head 重训

- [ ] 实现 rollout 数据收集脚本（存 z_t, a_t, task_success）
- [ ] 构造正负样本 dataset
- [ ] 只微调 Energy Head MLP（冻结 Mamba backbone）
- [ ] 训练并保存新的 cewm checkpoint

### Phase 3：验证

- [ ] 用新 Energy Head 跑 CE-AIS eval（1000 chains, seed=42）
- [ ] 对比 frozen_flower baseline
- [ ] 分析 diagnostics：energy_before/after 是否和 task success 正相关
- [ ] 如果 L5 有提升 → 成功，进入全面实验
- [ ] 如果仍无提升 → 尝试方案二/三

### Phase 4：全面实验（论文数据）

- [ ] Clean ABC→D 完整结果（L1-L5, avg_len, latency, diagnostics）
- [ ] OOD experiments（mild/medium physics + visual）
- [ ] Recovery subset 分析
- [ ] Ablation studies
- [ ] 多 seed 统计显著性

## 4. 关键文件路径

| 用途 | 路径 |
|------|------|
| CE-AIS 核心 config | `configs/base.yaml` |
| 激进调参 config | `configs/aggressive_steering.yaml` |
| Topology 决策逻辑 | `src/dual_stream/topology.py` |
| Energy Model | `src/world_model/ce_wm.py` |
| Energy Head | `src/world_model/energy_head.py` |
| Langevin Steering | `src/steering/langevin.py` |
| Bilateral Gating | `src/steering/bilateral_gating.py` |
| Training Loss | `src/training/losses.py` |
| Eval 脚本 | `scripts/run_paper_experiments.py` |
| Multi-seed 汇总 | `scripts/summarize_multi_seed.py` |
| Baseline 结果 | `results/flower_official_1000chains/merged_results.json` |
| Multi-seed eval | `results/multi_seed_eval/` |
| CE-AIS 激进测试 | `results/ce_ais_aggressive/` |

## 5. 进展日志

| 日期 | 内容 |
|------|------|
| 2026-06-08 | 完成 FLOWER eval pipeline 修复，跑出 4.281 avg_len baseline |
| 2026-06-09 | 发现 calvin_env 版本问题（button fix + action copy fix），升级到 797142c |
| 2026-06-09 | 确认 RandomShiftsAug 导致结果不确定性，去除后结果固定 |
| 2026-06-09 | 启动 multi_seed_eval（50 seeds, 16 并行进程） |
| 2026-06-10 | CE-AIS 激进调参测试：L5 从 76%→70%，反而变差 |
| 2026-06-10 | 诊断：energy 降低但任务失败 → CE-WM energy 和 task success 不对齐 |
| 2026-06-11 | 制定改进计划：方案一（rollout 数据重训 Energy Head）为首选 |
