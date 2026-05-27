# CE-AIS 组会实验汇报材料

> 位置：`/data0/yejinxuan/ce-ais/docs/CE_AIS_GROUP_MEETING_EXPERIMENT_REPORT_ZH.md`  
> 用途：组会 PPT 内容整理，可按章节直接拆成 slides。

---

## 1. 汇报主线

本轮实验的目标不是重新训练或微调 VLA，而是验证：

> **在冻结强 VLA 策略的前提下，CE-AIS 能否作为一个测试时动作空间校正模块，在不破坏 clean 性能的同时，对部分 OOD 场景提供恢复能力。**

当前最重要的结论：

1. **Clean 条件下成立**：CE-AIS 在 100 条 CALVIN 官方任务链上基本不伤 FLOWER，且 L1-L4 与平均完成数略有提升。
2. **OOD 条件下是混合结果**：visual mild / visual severe / physics severe 有小幅收益，但 physics、camera 与 visual medium 不稳定。
3. **安全 steering 框架有效但还不够“会拒绝”**：诊断显示动作偏移很小、能量稳定下降，但 accept rate 接近 99.6%–99.9%，说明当前模块更像“小步动作校正器”，还不是足够强的 abstentive verifier。
4. **后续不建议继续大规模烧测试**：当前结果已经能支撑一个谨慎但清晰的汇报结论；下一步应聚焦改进 accept/reject 与 uncertainty abstention，而不是继续全量扫实验。

---

## 2. 背景：为什么做 CE-AIS

### 2.1 问题背景

Vision-Language-Action（VLA）模型可以根据语言指令和视觉观测直接输出机器人动作，但在机器人任务中常见两个问题：

- **长时程任务错误累积**：前一个子任务的小偏差会影响后续任务链。
- **分布外环境不稳定**：光照、相机、物理参数变化后，策略动作可能偏离训练分布。

传统做法通常需要：

- 重新微调 VLA；
- 加入更多 OOD 数据；
- 设计任务特定恢复策略。

本项目希望避免这些依赖，提出一个 **model-agnostic 的测试时动作修正层**。

### 2.2 CE-AIS 的核心想法

CE-AIS（Causal-Energy Active Inference Steering）在测试时保持：

- VLA 冻结；
- Encoder 冻结；
- CE-WM 冻结；
- 只在动作空间中搜索更低能量的动作。

核心流程：

```text
观测 o_t + 语言指令
        │
        ▼
冻结 VLA / FLOWER 输出动作 a_vla
        │
        ▼
Encoder 编码观测为 latent z
        │
        ▼
CE-WM 评估 E(z, a)
        │
        ▼
Safe CE-AIS 在 trust-region 内修正动作 a*
        │
        ▼
Accept / Reject：若安全且能量下降则执行 a*，否则回退 a_vla
```

一句话概括：

> **CE-AIS 不是替换 VLA，而是在 VLA 输出动作后做一个冻结的、测试时的动作审查与小幅恢复控制。**

---

## 3. 数据集：CALVIN task_ABC_D

### 3.1 CALVIN 是什么

CALVIN（Composing Actions from Language and Vision）是一个面向语言条件机器人操作的长时程基准。它要求机器人根据自然语言指令连续完成多个桌面操作任务，例如：

- 打开/关闭抽屉；
- 移动滑门；
- 开关灯；
- 推动或旋转彩色积木；
- 抓取、抬起、堆叠积木。

本项目使用的是：

```text
data/task_ABC_D
```

即 **ABC→D 跨环境泛化协议**：

- 训练环境来自 A/B/C；
- 测试环境为 D；
- 因此它比单环境 D→D 更能测试泛化能力。

### 3.2 数据规模

当前项目中使用的 CALVIN ABC→D 数据：

| 项目 | 内容 |
|---|---|
| 数据目录 | `data/task_ABC_D` |
| 训练集 | 约 147 episodes |
| 训练帧数 | 约 179 万帧 |
| 原始格式 | 每帧一个 `.npz` |
| 加速格式 | `data/calvin_mmap/training` |
| 图像分辨率 | `rgb_static`: 200×200 |
| 任务形式 | 每条 evaluation chain 最多 5 个连续语言任务 |

为了训练速度，本项目把原始 `.npz` 转换为 mmap 格式：

```text
data/calvin_mmap/training
```

mmap 后只保留本项目需要的字段，避免每个 batch 频繁打开和解压大量 `.npz` 文件。

### 3.3 每帧使用的主要字段

| 字段 | shape | 作用 |
|---|---:|---|
| `rgb_static` | `(200, 200, 3)` | 第三人称静态 RGB 视觉观测 |
| `depth_static` | `(200, 200)` | 静态视角深度图，提供几何信息 |
| `robot_obs[:7]` | `(7,)` | TCP 位姿 + 夹爪状态 |
| `rel_actions[:7]` | `(7,)` | 相对动作，作为 VLA/CE-AIS 的动作空间 |

没有使用 `scene_obs`，因为它依赖外部真实状态，在真实机器人部署中通常不可得。

### 3.4 评估指标含义

CALVIN 使用长链成功率：

| 指标 | 含义 |
|---|---|
| L1 / `chain_1` | 完成第 1 个子任务的任务链比例 |
| L2 / `chain_2` | 连续完成前 2 个子任务的比例 |
| L3 / `chain_3` | 连续完成前 3 个子任务的比例 |
| L4 / `chain_4` | 连续完成前 4 个子任务的比例 |
| L5 / `chain_5` | 完整完成 5 个子任务链的比例 |
| `avg_completed_tasks` | 每条任务链平均完成的子任务数 |
| `avg_latency_ms` | 单步策略推理平均延迟 |

例如：

```text
L1=90%, L5=72%, avg_completed_tasks=3.98
```

表示 100 条任务链中：

- 90 条至少完成第 1 个任务；
- 72 条完整完成 5 个任务；
- 平均每条链完成 3.98 个任务。

---

## 4. 模型与方法

### 4.1 冻结 VLA：FLOWER

本轮实验使用 FLOWER 作为主 VLA backbone：

| 项目 | 内容 |
|---|---|
| VLA 类型 | `flower` |
| checkpoint | `data/flower_calvin_abc` |
| 代码路径 | `external/flower_vla_calvin` |
| 评估方法名 | `frozen_flower` |
| 是否训练 | 否，完全冻结 |

FLOWER 作为强 CALVIN baseline，已经能在 clean ABC→D 上达到较高 L1-L5，因此 CE-AIS 的首要要求是：

> **不能破坏 FLOWER clean 性能。**

### 4.2 Encoder

Encoder 用于把视觉和机器人状态编码到紧凑 latent：

| 参数 | 值 |
|---|---:|
| backbone | ResNet18 |
| 输入 RGB | 200×200 |
| 输入 pose | 7 维 |
| latent dim | 128 |
| checkpoint | `checkpoints/encoder_epoch0044.pt` |
| 测试时是否冻结 | 是 |

输入：

```text
rgb_static + depth_static + robot_obs[:7]
```

输出：

```text
z ∈ R^128
```

### 4.3 CE-WM：Causal Energy World Model

CE-WM 学习一个能量函数：

```text
E(z, a)
```

低能量表示动作更接近训练数据中的合理动作，高能量表示动作可能不合理。

主要结构参数：

| 参数 | 值 |
|---|---:|
| backbone | Mamba-style causal sequence model |
| d_model | 640 |
| d_state | 64 |
| n_layers | 32 |
| action_dim | 7 |
| latent_dim | 128 |
| dropout | 0.1 |
| 负样本数 K | 5 |

本轮最终使用 checkpoint：

```text
checkpoints_calibrated_cewm/cewm_epoch0033.pt
```

训练日志显示 CE-WM 训练稳定：

```text
loss ≈ 0.024
margin ≈ +5.75
|E| ≈ 2.88
grad ≈ 0.2~0.3
```

说明没有出现旧版本中的 margin 爆炸或 NCE 坍塌。

### 4.4 Safe CE-AIS 配置

主实验使用：

```text
configs/safe_balanced.yaml
```

关键 steering 参数：

| 参数 | 值 | 含义 |
|---|---:|---|
| `n_steps` | 2 | 每步动作修正迭代次数 |
| `step_size` | 0.006 | Langevin / EFE 动作搜索步长 |
| `noise_scale` | 0.0 | 关闭随机探索，保证确定性和稳定性 |
| `kl_weight` | 25.0 | 约束动作靠近 VLA prior |
| `action_delta_max` | 0.08 | 每维动作最大偏移 trust-region |
| `enable_accept_reject` | true | 能量没改善则回退 VLA 动作 |
| `accept_energy_margin` | 0.0001 | 需要满足能量下降阈值 |
| `lambda_max` | 0.4 | 最大门控强度 |
| `mc_samples` | 5 | MC dropout 不确定性采样数 |

CE-AIS 诊断指标包括：

- accepted / rejected rate；
- action delta 平均幅度；
- energy_before / energy_after；
- uncertainty；
- gating lambda。

---

## 5. 实验设置

### 5.1 Clean 主实验

目的：验证 CE-AIS 是否破坏 strong VLA prior。

| 项目 | 设置 |
|---|---|
| 数据集 | CALVIN ABC→D validation |
| sequence source | official |
| 任务链数 | 100 chains |
| 每条链长度 | 5 tasks |
| 最大步数 | 360 steps/task |
| 对比方法 | `frozen_flower` vs `ce_ais` |
| CE-AIS 配置 | `safe_balanced.yaml` |
| CE-WM | `cewm_epoch0033.pt` |

命令输出：

```text
results/final_balanced_cewm33_clean_100/main_experiment.json
logs/final_balanced_cewm33_clean_100.log
```

### 5.2 OOD 全范围实验

目的：测试物理、视觉、相机扰动下的鲁棒性。

| OOD 类型 | mild | medium | severe |
|---|---|---|---|
| physics | mass×1.2, friction×0.8 | mass×1.5, friction×0.5 | mass×2.0, friction×0.3 |
| visual | brightness=0.8, noise=0.02 | brightness=0.6, noise=0.05 | brightness=0.5, noise=0.10 |
| camera | offset=0.01 | offset=0.03 | offset=0.05 |

实验设置：

| 项目 | 设置 |
|---|---|
| 每个 OOD severity | 100 episodes |
| 每条 episode | 最多 5 tasks |
| 对比方法 | `frozen_flower` vs `ce_ais` |
| CE-AIS 配置 | `safe_balanced.yaml` |
| CE-WM | `cewm_epoch0033.pt` |

命令输出：

```text
results/final_balanced_cewm33_ood_all_100/ood_experiment.json
logs/final_balanced_cewm33_ood_all_100.log
```

---

## 6. Clean 主实验结果

### 6.1 L1-L5 与平均完成数

| 方法 | L1 | L2 | L3 | L4 | L5 | Avg completed | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen FLOWER | 90.0 | 84.0 | 79.0 | 73.0 | 72.0 | 3.98 | 41.4 ms |
| CE-AIS | 92.0 | 87.0 | 80.0 | 74.0 | 71.0 | 4.04 | 244.1 ms |
| Δ | +2.0 | +3.0 | +1.0 | +1.0 | -1.0 | +0.06 | +202.7 ms |

### 6.2 clean 结果解读

Clean 结果是本轮最重要的正结果：

- CE-AIS 没有破坏 FLOWER；
- L1-L4 均略有提升；
- L5 小幅下降 1 个点；
- 平均完成任务数从 `3.98` 提升到 `4.04`；
- 0-task failure 从 10 条减少到 8 条。

完成任务数分布：

| 方法 | 0 task | 1 task | 2 tasks | 3 tasks | 4 tasks | 5 tasks |
|---|---:|---:|---:|---:|---:|---:|
| Frozen FLOWER | 10 | 6 | 5 | 6 | 1 | 72 |
| CE-AIS | 8 | 5 | 7 | 6 | 3 | 71 |

这说明 CE-AIS 更主要的作用是减少早期失败和轻微提升长链稳定性，而不是显著增加完整 5-task 成功数。

### 6.3 clean 诊断

CE-AIS clean 诊断：

| 指标 | 值 |
|---|---:|
| accepted_rate | 0.9974 |
| rejected_rate | 0.0026 |
| action_delta_inf_mean | 0.00128 |
| energy_before_mean | -1.0201 |
| energy_after_mean | -1.0322 |
| uncertainty_mean | 0.0238 |
| gating_lambda_mean | 0.3884 |

解读：

- 动作修正幅度极小，说明 CE-AIS 保持在 FLOWER policy manifold 附近；
- energy_after 更低，说明 CE-WM 确实引导动作向低能量方向移动；
- rejected_rate 很低，说明当前配置几乎总是接受小幅修正。

---

## 7. OOD 实验结果

### 7.1 全 OOD 汇总表

表中数值格式：`L1 / L2 / L3 / L4 / L5 / Avg`。

| OOD | Frozen FLOWER | CE-AIS | Δ L1 | Δ Avg | 判断 |
|---|---:|---:|---:|---:|---|
| physics/mild | 16 / 2 / 1 / 0 / 0 / 0.19 | 16 / 0 / 0 / 0 / 0 / 0.16 | +0 | -0.03 | 负向 |
| physics/medium | 6 / 1 / 0 / 0 / 0 / 0.07 | 5 / 1 / 0 / 0 / 0 / 0.06 | -1 | -0.01 | 轻微负向 |
| physics/severe | 7 / 1 / 0 / 0 / 0 / 0.08 | 8 / 1 / 0 / 0 / 0 / 0.09 | +1 | +0.01 | 轻微正向 |
| visual/mild | 31 / 9 / 2 / 0 / 0 / 0.42 | 32 / 10 / 3 / 0 / 0 / 0.45 | +1 | +0.03 | 正向 |
| visual/medium | 34 / 11 / 4 / 0 / 0 / 0.49 | 28 / 11 / 3 / 0 / 0 / 0.42 | -6 | -0.07 | 明显负向 |
| visual/severe | 30 / 10 / 3 / 0 / 0 / 0.43 | 30 / 10 / 3 / 1 / 0 / 0.44 | +0 | +0.01 | 轻微正向 |
| camera/mild | 30 / 12 / 4 / 0 / 0 / 0.46 | 30 / 9 / 4 / 1 / 0 / 0.44 | +0 | -0.02 | 轻微负向 |
| camera/medium | 31 / 10 / 4 / 0 / 0 / 0.45 | 30 / 10 / 4 / 0 / 0 / 0.44 | -1 | -0.01 | 轻微负向 |
| camera/severe | 31 / 12 / 4 / 0 / 0 / 0.47 | 28 / 11 / 4 / 0 / 0 / 0.43 | -3 | -0.04 | 负向 |

### 7.2 OOD 结果解读

OOD 结果不是全面提升，而是混合结果。

正向信号：

- `visual/mild`: avg `0.42 → 0.45`；
- `visual/severe`: avg `0.43 → 0.44`，并出现 L4 `0 → 1%`；
- `physics/severe`: avg `0.08 → 0.09`。

负向信号：

- `visual/medium`: L1 `34% → 28%`，avg `0.49 → 0.42`，这是最明显的失败点；
- `camera/severe`: L1 `31% → 28%`，avg `0.47 → 0.43`；
- `physics/mild`: avg `0.19 → 0.16`。

因此，汇报中不要说：

> CE-AIS 全面提升 OOD 鲁棒性。

更准确的表述是：

> CE-AIS 在 clean 条件下保持并略微提升强 VLA 表现；在 OOD 条件下呈现选择性收益，尤其在部分 visual 和 severe stress test 中有恢复迹象，但当前 accept/reject 机制仍不足以稳定处理所有扰动类型。

### 7.3 OOD 诊断分析

各 OOD 场景中 CE-AIS 的诊断特征非常一致：

| 指标范围 | 观察 |
|---|---|
| accepted_rate | 约 0.996–0.999 |
| rejected_rate | 约 0.001–0.004 |
| action_delta_inf_mean | 约 0.00128–0.00130 |
| gating_lambda_mean | 约 0.384–0.391 |
| energy_after_mean | 均低于 energy_before_mean |

说明：

1. CE-AIS 一直能找到更低能量动作；
2. 动作偏移很小；
3. 但 accept/reject 几乎总是接受，缺少强 abstention；
4. 对于某些 OOD，CE-WM 的低能量方向不一定等价于任务成功方向。

这解释了为什么 clean 表现稳定，但 OOD 结果混合。

---

## 8. 训练与 checkpoint 选择

### 8.1 为什么重新训练 CE-WM

早期 CE-WM 出现过能量 margin 爆炸与坍塌风险：

- margin 可能冲到几十甚至几百；
- loss 接近 `ln(1+K)` 时表示 NCE 分类接近随机；
- 这样的能量场不适合作为动作梯度。

因此本轮引入 calibrated CE-WM loss：

```text
L = L_NCE
  + energy_reg_weight * mean(E_pos^2 + E_neg^2)
  + margin_upper_weight * relu(margin - target_margin)^2
  + margin_lower_weight * relu(min_margin - margin)^2
```

关键训练参数：

| 参数 | 值 |
|---|---:|
| learning_rate | 1e-4 |
| cewm_batch_size | 256 per GPU |
| neg_sample_ratio | 5 |
| mixed_precision | bf16 |
| energy_reg_weight | 1e-4 |
| target_margin | 5.0 |
| min_margin | 1.0 |
| margin_upper_weight | 1e-2 |
| margin_lower_weight | 1.0 |

### 8.2 checkpoint sweep 结论

做过小规模 checkpoint 对比后，最终选择：

```text
checkpoints_calibrated_cewm/cewm_epoch0033.pt
```

原因：

- clean 20-chain 中提升明显；
- visual OOD 中相对稳；
- 训练指标已经平台期；
- margin 稳定在约 5.75，没有爆炸。

---

## 9. 可以放进 PPT 的核心结论

### 9.1 一页总结

> 本轮实验验证了 Safe CE-AIS 可以在冻结 FLOWER VLA 的前提下，通过低幅度动作空间 steering 保持 clean 长链性能，并带来轻微提升。100-chain clean 结果中，CE-AIS 将 L1/L2/L3/L4 从 90/84/79/73 提升到 92/87/80/74，平均完成任务数从 3.98 提升到 4.04。但 OOD 结果显示当前 CE-WM 能量下降并不总是对应任务成功，physics/camera 以及 visual medium 仍存在负向 case。因此下一步应改进 abstention 和 accept/reject，而不是继续扩大实验规模。

### 9.2 方法贡献表述

可以这样讲：

1. **模型无关**：CE-AIS 接在 VLA action output 后面，不依赖 FLOWER 内部结构。
2. **测试时修正**：不更新 VLA 参数，只搜索动作张量。
3. **安全约束**：trust-region 限制动作偏移，accept/reject 支持回退。
4. **可诊断**：输出干预率、能量变化、不确定性、动作偏移。
5. **实验证明 clean preservation**：在强 VLA 上没有明显退化，并略微提升平均完成数。

### 9.3 当前局限

需要诚实说明：

1. OOD 不是全面提升；
2. accept/reject 目前太宽松，accepted_rate 过高；
3. CE-WM 低能量方向和任务成功方向并不总是完全一致；
4. 推理延迟明显增加：clean 中从约 41 ms 增加到 244 ms；
5. 目前主要验证 FLOWER，后续还需要接更多 VLA/action policy 验证 model-agnostic claim。

---

## 10. 后续工作建议

### 10.1 短期：不要继续大规模跑实验

当前结果已经足够用于组会汇报。继续跑 physics/camera 大 sweep 性价比低，因为主要结论已经清楚：

- clean 稳定；
- OOD 混合；
- 需要改进 abstention。

### 10.2 方法改进方向

优先做：

1. **更严格 accept/reject**
   - 提高 `accept_energy_margin`；
   - 或使用相对能量下降比例；
   - 避免微小能量下降也被接受。

2. **hard uncertainty abstention**
   - 设定 `hard_uncertainty_threshold`；
   - OOD 不确定性过高时直接执行 VLA prior。

3. **任务成功相关的 energy calibration**
   - 当前 CE-WM 学的是动作合理性，不一定直接对应任务完成；
   - 后续可以加入 failure/recovery subset 或 goal-conditioned energy。

4. **降低延迟**
   - 当前 CE-AIS 单步约 244 ms；
   - 可考虑减少 CE-WM 调用、缓存 encoder、优化 finite-difference 或尝试安全的 compile 策略。

### 10.3 下一轮实验建议

如果后续要继续改进，不建议直接再大跑。推荐：

1. 先在 20-chain clean + 50-episode visual 上小规模验证；
2. 只在 small-scale 正向后再跑 100-chain clean；
3. OOD 只选最能说明问题的场景，例如 visual/mild、visual/medium、camera/severe；
4. 每次只改一个配置变量，避免无法归因。

---

## 11. PPT 建议结构

可以按以下 10 页组织：

1. **Motivation**：VLA 长链/OOD 问题，为什么需要测试时动作修正。
2. **Dataset**：CALVIN ABC→D、5-task chain、L1-L5 指标。
3. **Baseline**：FLOWER frozen VLA，clean 已经很强。
4. **Method**：CE-AIS 总体框架图。
5. **Safe CE-AIS**：trust-region + accept/reject + uncertainty gating。
6. **Training**：Encoder + CE-WM，calibrated NCE loss。
7. **Clean Results**：100-chain clean 表格，强调 non-degradation。
8. **OOD Results**：9 个 OOD 场景表格，强调 mixed robustness。
9. **Diagnostics**：动作偏移小、energy 降低、accepted_rate 太高。
10. **Conclusion & Next Step**：当前结论 + 下一步改进 abstention。

---

## 12. 一句话结论

> Safe CE-AIS 已经能在冻结强 VLA 的情况下实现 clean 长链性能保持与小幅提升，但当前 OOD 结果仍是选择性收益；下一步的关键不是继续扩大实验，而是让 CE-AIS 在不确定或错误能量场下更会“拒绝干预”。
