# CE-AIS 论文级改进方案：从动作偏转器升级为通用安全动作审查与恢复控制器

## 0. 结论先行

当前实验问题不是 CE-AIS 方向错误，而是现有实现还没有完全兑现论文中 `CE-AIS = 冻结 VLA + 因果能量裁判 + 主动推理式安全动作选择` 的架构承诺。

现有 clean FLOWER 结果：

```text
frozen_flower  L1=95.0%  L2=92.0%  L3=87.0%  L4=79.5%  L5=71.0%  Lat=24.6ms
ce_ais         L1=94.0%  L2=90.5%  L3=84.0%  L4=80.0%  L5=73.5%  Lat=167.2ms
```

现有 physics OOD severe 结果：

```text
physics OOD: mass x2.0, friction x0.3
frozen_flower  L1=8.0%, avg_completed=0.09
ce_ais         L1=7.0%, avg_completed=0.07
```

这些结果说明：

1. CE-AIS 不是完全无效：clean L5 从 71.0% 到 73.5%，说明它确实救回了一部分长链后段失败。
2. CE-AIS 当前也会破坏强基座：L1/L2/L3 下降，说明每步动作偏转正在干扰 FLOWER 的高质量 action manifold。
3. Severe physics OOD 太强，当前 frozen 与 CE-AIS 都崩，不适合直接作为主结论。
4. CE-WM 训练存在能量尺度爆炸和后期坍塌，导致其作为动作梯度场不可靠。

因此后续方向不是放弃 CE-AIS，而是把 CE-AIS 从“每步强制动作偏转器”升级为：

> **一个 model-agnostic、parameter-frozen、可拒绝介入的安全动作能量审查与恢复控制器。**

这不会削弱论文创新点，反而会让创新点更强：原始设想强调 zero-gradient test-time steering；改进后进一步具备 trust-region、abstention、goal-conditioned energy、failure recovery 和跨动作模型通用性，更符合顶会论文对稳定性、可解释性和可复现实验的要求。

---

## 1. 对论文创新点的影响

### 1.1 改动大不大？

从论文核心概念看，改动不大；从工程实现看，需要分阶段修正。

保留不变的核心创新：

- VLA 基座冻结，不做 test-time finetuning。
- CE-WM 冻结，不在测试时更新参数。
- CE-AIS 工作在 action-space，而不是更新视觉编码器、LoRA 或策略权重。
- 外层仍然通过统一 `VLAAdapter.predict(obs, instruction) -> Tensor[B,T,7]` 接多个动作模型。
- 主线仍然是主动推理 / expected free energy / energy-based world model。

需要修正的实现形式：

- 从“每步直接 Langevin 偏转动作”改为“候选动作审查 + trust-region 小步 steering + accept/reject”。
- 从“只判断动作是否像训练数据”改为“任务条件下的风险、进度和不确定性联合评分”。
- 从“永远介入”改为“有把握才介入，没把握就回退 VLA”。
- 从“单一 clean demo NCE 能量”改为“包含失败、stuck、OOD、不同 policy rollout 的通用 CE-WM”。

### 1.2 创新点会不会变弱？

不会。原始 CE-AIS 的理论表达是：

```text
Frozen VLA + Frozen Energy World Model + Test-time Action Steering
```

改进后的表达是：

```text
Frozen VLA + Frozen Goal-conditioned Energy Verifier + Safe Active-Inference Action Selection
```

这比原始版本更强，因为它增加了三个顶会审稿人很在意的性质：

1. **Non-degradation safety**：强基座 clean 场景不应被无意义破坏。
2. **Abstention under uncertainty**：世界模型不确定时不盲目输出错误梯度。
3. **Cross-model generality**：CE-WM 不是给 FLOWER 单独训练的补丁，而是从多动作源 rollout 学到的通用 failure/risk verifier。

论文叙事建议从：

```text
CE-AIS always improves frozen VLA.
```

改成：

```text
CE-AIS is a frozen, model-agnostic test-time action verifier and recovery controller. It preserves strong VLA behavior on clean data, selectively intervenes under high-risk or low-progress states, and improves long-horizon robustness and recoverability under distribution shifts.
```

中文表达：

> CE-AIS 不是无条件替代 VLA，而是在冻结 VLA 的动作先验周围进行安全审查、低风险候选选择和必要时的恢复控制；它的价值不在于对所有 clean step 都强行优化，而在于让强动作模型在长链、失败边界和分布偏移中少犯不可恢复错误。

---

## 2. 当前失败原因诊断

### 2.1 CE-WM 训练能量尺度失控

你之前的 CE-WM 训练日志：

```text
CE-WM epoch 1/200  - loss=0.0536, margin=+25.8965
CE-WM epoch 15/200 - loss=0.0014, margin=+61.3835
CE-WM epoch 16/200 - loss=0.0046, margin=+72.2405
CE-WM epoch 18/200 - loss=0.0379, margin=+310.9257
CE-WM epoch 19/200 - loss=0.0027, margin=+280.5909
CE-WM epoch 24/200 - loss=37.6184, margin=+54.5314
CE-WM epoch 25/200 - loss=2.3050, margin=+0.0148
CE-WM epoch 27/200 - loss=1.7920, margin=+0.1228
CE-WM epoch 35/200 - loss=1.7918, margin=+0.0310
```

关键判断：

- `loss ≈ 0` 且 `margin` 持续变大，不代表能量模型更好，而可能说明 InfoNCE 通过无约束放大 energy scale 获得近似零损失。
- `margin +310` 是危险信号：这种能量场对动作的梯度很可能数值极端，不适合作为稳定 steering field。
- 后期 `loss ≈ 1.7918` 接近 `ln(6)`，如果训练是 1 个正样本 + 5 个负样本，这通常表示模型退化到随机分类或能量坍塌。
- epoch 18 之后的 checkpoint 不应直接用于 steering；即使 loss 低，也可能是坏的能量景观。

根因：

1. InfoNCE 没有能量尺度正则，margin 可以无限增大。
2. 没有 target margin，模型不知道“多大差距已经够了”。
3. 负样本过于容易，导致 early epoch 很快靠尺度拉开。
4. 缺少能量校准指标，比如能量均值、方差、梯度范数、OOD 分离度、success correlation。
5. 当前 CE-WM 学的是 action plausibility，不是 task-conditioned success/risk。

### 2.2 每步直接动作偏转破坏强基座

FLOWER 本身 clean L1 已经 95%，L5 已经 71%。这类强 action model 的动作 chunk 有隐式结构：靠近、对齐、闭合夹爪、抬升、转移、释放。当前 CE-AIS 每步对输出动作做 Langevin 偏转，会造成：

- 单步看似很小，累计几十步后轨迹偏离 FLOWER policy manifold。
- 对早期简单任务造成不必要扰动，所以 L1/L2/L3 下降。
- 偶尔在后段失败边界救回一些轨迹，所以 L5 小幅提升。

这解释了现有结果：

```text
L1/L2/L3 drop, L5 slightly improve
```

这不是理想的 CE-AIS 行为。理想行为应该是：

```text
clean: mostly abstain, no degradation
failure-prone states: selective intervene
OOD/recovery: improve or safely abstain
```

### 2.3 Gating 机制没有形成可靠安全保护

论文中 bilateral epistemic gating 的设计目标是：世界模型不确定时降低引导强度，回退到 VLA。

当前实现风险：

- `DualStreamTopology.reset()` 只 reset 了 VLA，没有 reset gating history。
- gating 使用在线历史均值，强 OOD 下可能把异常不确定性纳入新均值，导致门控失去保护性。
- 没有 hard threshold abstention；即使 uncertainty 高，只要 gating 没降到 0，仍可能产生有害 steering。
- 没有 accept/reject；energy 没变好或动作偏移太大时仍然输出修改动作。

### 2.4 Severe OOD 不适合作为第一主结果

当前 physics OOD：

```text
mass_scale = 2.0
friction_scale = 0.3
```

这对 CALVIN 操作是很强的 dynamics shift。frozen FLOWER 只有 8% L1，说明该设置可能已经超过基座可恢复区间。

论文主结果不应只放最强 OOD。应采用 severity sweep：

```text
mild:   mass 1.2, friction 0.8
medium: mass 1.5, friction 0.5
severe: mass 2.0, friction 0.3
```

主 claim 放 mild/medium，severe 作为 stress test 和 abstention 分析。

---

## 3. 总体改进目标

后续 CE-AIS 应满足四个目标：

### 3.1 Clean preservation

在强 frozen model 上，clean 场景不明显掉点：

```text
L1/L2/L3 不应显著低于 frozen
L5 可小幅提升或持平
```

这是所有后续 OOD claim 的前提。如果 clean 明显掉点，审稿人会认为 CE-AIS 是有害扰动器。

### 3.2 Selective intervention

CE-AIS 不应每步都修正动作，而应在以下场景介入：

- energy 明显异常；
- uncertainty 可控且模型有把握；
- progress 停滞；
- VLA 动作重复或 stuck；
- 当前状态属于长链后段高失败风险区域。

### 3.3 Model-agnostic generality

CE-AIS 不应为每个动作模型单独训练一个专用补丁。更好的做法是训练一个通用 CE-WM：

```text
expert demos + FLOWER rollouts + RoboFlamingo rollouts + RoboVLMs rollouts + CE-AIS failed rollouts
```

训练标签围绕 outcome，而不是围绕模型身份：

```text
success / failure / stuck / wrong-task / no-progress / high-jerk / collision-risk
```

### 3.4 Recovery-oriented improvement

L5 大幅提升不应依赖每步微调，而应依赖 failure recovery：

- 找到 frozen 容易失败的 chains；
- 检测 stuck 或低进度状态；
- 在高风险片段中启用 CE-AIS；
- 让 long-horizon 后段恢复，而不是在所有 clean step 上扰动。

---

## 4. 架构改进方案

## 4.1 Safe CE-AIS：从强制偏转到可拒绝介入

### 当前形式

```text
a_vla = VLA(obs, instruction)
a_star = Langevin(E, a_vla)
execute(a_star)
```

### 改进形式

```text
a_vla = VLA(obs, instruction)
a_candidate = CE-AIS propose / rerank / small steer

if safe_accept(a_candidate, a_vla):
    execute(a_candidate)
else:
    execute(a_vla)
```

### accept/reject 条件

建议加入：

```text
1. energy_after < energy_before - energy_margin
2. ||a_star - a_vla||_inf <= action_delta_max
3. uncertainty <= uncertainty_threshold
4. predicted_progress_after >= predicted_progress_before
5. action smoothness / jerk 不恶化
```

任何条件不满足，就回退 frozen VLA action。

### trust-region 限制

限制 CE-AIS 只能在 VLA 动作附近小范围修改：

```text
delta = clamp(a_star - a_vla, -delta_max, +delta_max)
a_safe = a_vla + delta
```

建议初始值：

```yaml
action_delta_max: 0.03 ~ 0.08
```

具体要按 CALVIN action scale 扫描。

### 预期收益

- clean L1/L2/L3 不再明显掉点；
- CE-AIS 的负作用被 accept/reject 截断；
- 论文可以报告 abstention rate 和 harmful intervention rejection rate，增强可解释性。

---

## 4.2 Reranking-first：优先候选打分，谨慎梯度 steering

### 当前风险

直接梯度 steering 容易把动作推出 VLA policy manifold。

### 推荐改法

先不要每步直接改动作，而是在 VLA 原动作附近生成候选：

```text
A = {a_vla, a_vla + eps_1, ..., a_vla + eps_K}
a_best = argmin score(a)
```

score：

```text
score(a) =
    E_physics(z, a)
  - λ_progress * Progress(z, a, instruction)
  + β * ||a - a_vla||²
  + γ * Uncertainty(z, a)
```

如果 `a_best` 不明显优于 `a_vla`，就执行 `a_vla`。

### 候选生成方式

第一阶段可以很简单：

- Gaussian small perturbations；
- gripper 维度单独少扰动或不扰动；
- 只扰动 xyz / rotation 中低风险维度；
- 保留原 FLOWER chunk 的时序结构。

第二阶段再加入 Langevin refinement：

```text
reranking -> accepted candidate -> small-step Langevin -> accept/reject
```

### 预期收益

- 更稳；
- 更通用；
- 更符合“能量审查器”定位；
- 比直接 steering 更容易避免 clean 掉点。

---

## 4.3 Goal-conditioned CE-WM：从动作合法性到任务条件风险/进度评估

### 当前 CE-WM 问题

当前 CE-WM 主要学习：

```text
E(z, a): action 是否像训练数据中的合法动作
```

这不足以做 active inference，因为它不知道任务目标。

### 改进目标

改成：

```text
E(z, a, instruction)
```

或者更完整：

```text
E(z_t, a_{t:t+H}, instruction, progress_state)
```

### 训练数据

正样本：

- expert demo successful windows；
- frozen FLOWER 成功 rollout windows；
- 后续 RoboFlamingo / RoboVLMs 成功 rollout windows。

负样本：

- action perturbation negatives；
- failed rollout 前 N 步 windows；
- stuck windows；
- wrong-task action windows；
- high-jerk / oscillation windows；
- CE-AIS 改坏的 windows。

### 训练目标

不再只用 InfoNCE。建议采用多任务目标：

```text
L = L_rank + L_bce_success + L_margin + L_energy_reg + L_grad_reg
```

其中：

- `L_rank`：成功 window 能量低于失败 window；
- `L_bce_success`：预测该候选动作是否带来 progress/success；
- `L_margin`：使用目标 margin，避免无限拉大；
- `L_energy_reg`：限制 energy scale；
- `L_grad_reg`：限制对 action 的梯度范数，避免 steering field 爆炸。

---

## 4.4 Progress-aware active inference

论文中 Expected Free Energy 不应只体现 physical energy，还应体现 goal achievement。

建议 CE-AIS 分数改成：

```text
G(a) = RiskEnergy(z, a)
     - λ_goal * Progress(z, a, instruction)
     + β * Deviation(a, a_vla)
     + γ * Uncertainty(z, a)
```

解释：

- `RiskEnergy`：动作是否物理风险高；
- `Progress`：动作是否让任务更接近成功；
- `Deviation`：不要偏离 VLA 先验太远；
- `Uncertainty`：世界模型不确定时减少介入。

这比当前单纯 energy minimization 更接近主动推理理论。

---

## 4.5 Stuck-triggered recovery controller

为了显著提升 L5，CE-AIS 不应主要用于所有步骤，而应用于 frozen VLA 的失败边界。

### stuck 检测信号

可以使用：

```text
1. 连续 N 步任务 reward/progress 不变
2. 末端位姿变化小
3. 动作重复度高
4. gripper 状态不合理
5. energy 持续升高
6. 当前子任务已超过历史成功步数分位数
```

### 触发策略

```text
if not stuck:
    execute VLA action
else:
    activate CE-AIS recovery mode
```

### recovery mode

- 增大候选数量 K；
- 允许稍大 trust-region；
- 使用 progress-aware reranking；
- 仍保留 accept/reject。

### 预期收益

这比每步修改更容易提升 L5，因为 L5 的主要问题是长链后段 stuck 和状态偏移，而不是每个 clean step 都需要修。

---

## 5. CE-WM 训练修正方案

## 5.1 不再用后期崩坏 checkpoint

根据当前日志，优先避免：

```text
cewm_epoch0018.pt 之后的 checkpoint
```

特别是：

```text
epoch 18: margin +310
后期: loss ≈ ln(6)
```

短期实验建议只比较：

```text
cewm_epoch0015.pt
cewm_epoch0016.pt
cewm_epoch0017.pt
```

但这只是临时止血，不是根本解决。

## 5.2 加 energy scale regularization

在 CE-WM loss 中加入：

```text
L_energy_reg = α * (E_pos².mean() + E_neg².mean())
```

目的：防止模型通过无限放大能量差来压低 InfoNCE loss。

建议初值：

```yaml
energy_reg_weight: 1e-4 ~ 1e-3
```

## 5.3 加 target margin，不允许 margin 无限增大

当前 margin 从 +25 到 +310，是不健康的。

建议将 margin 目标限制在：

```text
target_margin = 2 ~ 10
```

loss 示例：

```text
L_margin = (observed_margin - target_margin)^2
```

或者使用 hinge：

```text
max(0, target_margin - margin)
```

但不奖励 margin 超过 target 太多。

## 5.4 加 gradient norm regularization

因为 CE-WM 要作为 steering field，关键不是分类准确率，而是动作梯度是否平滑。

建议：

```text
L_grad = ||∂E/∂a||²
```

或只监控不直接训练，至少记录：

```text
grad_norm_mean
grad_norm_p95
grad_norm_max
```

如果梯度过大，即使 loss 好，也不能用于 CE-AIS。

## 5.5 训练日志必须新增诊断指标

每个 epoch 记录：

```text
loss
margin
energy_pos_mean
energy_pos_std
energy_neg_mean
energy_neg_std
energy_abs_mean
grad_norm_mean
grad_norm_p95
success_auc 或 pos_neg_auc
calibration_ece
```

checkpoint 选择标准不再是 loss 最低，而是：

```text
1. validation pos/neg AUC 高
2. margin 在合理区间
3. energy scale 稳定
4. grad norm 平滑
5. steering smoke 不伤 frozen
```

## 5.6 负样本要从 easy negative 升级为 hard negative

当前负样本如果只是随机扰动，很容易被模型分开，导致 margin 爆炸。

建议加入：

- VLA 失败动作；
- 任务错配动作；
- stuck 前动作；
- 轻微但危险的 gripper 错误；
- OOD 下 frozen 失败动作；
- CE-AIS 曾经改坏的动作。

这会让 CE-WM 学到真正对控制有用的能量边界。

---

## 6. 实验设计修正方案

## 6.1 Clean 主表：证明不伤强基座

clean 表不要强求大幅提升 L1。FLOWER L1 已经 95%，可提升空间只有 5%。

主目标：

```text
clean no degradation + L5/recovery modest gain
```

建议表格：

```text
Model          Method          L1    L2    L3    L4    L5    Lat(ms)
FLOWER         Frozen
FLOWER         +CE-AIS-safe
RoboFlamingo   Frozen
RoboFlamingo   +CE-AIS-safe
RoboVLMs       Frozen
RoboVLMs       +CE-AIS-safe
```

合格标准：

```text
L1/L2/L3 不显著下降
L5 持平或提升
latency 可解释
```

## 6.2 OOD 主表：使用 severity sweep

不要只放 severe physics。

建议：

```text
physics:
  mild:   mass 1.2, friction 0.8
  medium: mass 1.5, friction 0.5
  severe: mass 2.0, friction 0.3

visual:
  mild:   brightness 0.8, noise 0.02
  medium: brightness 0.6, noise 0.05
  severe: brightness 0.5, noise 0.10

camera:
  mild:   offset 0.01
  medium: offset 0.03
  severe: offset 0.05
```

主 claim 放 mild/medium；severe 用来展示 abstention 和 failure boundary。

## 6.3 Recovery 曲线：作为论文重点图

Recovery 更贴合 CE-AIS 的创新点。

建议画：

```text
x-axis: steps after perturbation
y-axis: success/progress/recovery score
methods: frozen, CE-AIS, CE-AIS w/o gating, CE-AIS w/o accept-reject
```

目标现象：

```text
frozen: drop 后恢复慢或不恢复
CE-AIS-safe: drop 较小或恢复更快
w/o gating: 可能过度干预
w/o accept-reject: 可能伤害 clean
```

## 6.4 Failure-subset evaluation

为了展示 L5 改善潜力，建议增加 diagnostic 评估：

1. 用 frozen FLOWER 跑 200/500 chains。
2. 记录 frozen 失败的 chain。
3. 在同一批失败 chain 上跑 CE-AIS。
4. 报告：

```text
Frozen-failed chains recovered by CE-AIS: X%
```

这不是替代官方主表，而是解释 CE-AIS 在哪些失败模式上有用。

## 6.5 Conditional long-horizon success

不要只看 L5，也看：

```text
P(task k succeeds | task 1..k-1 succeeded)
```

这能解释 CE-AIS 是否真正改善长链后段，而不是只靠随机性。

## 6.6 Intervention diagnostics

必须记录：

```text
intervention_rate
accept_rate
reject_rate
mean_action_delta
energy_before / after
uncertainty_mean
progress_score_before / after
success when intervened
success when abstained
```

这类指标能让审稿人相信 CE-AIS 是一个可解释控制器，而不是黑箱后处理。

---

## 7. 分阶段实施路线

## Stage 1：安全保护层，先止住 clean 掉点

目标：把当前 CE-AIS 从无保护 steering 改成 safe steering。

改动：

1. `DualStreamTopology.reset()` 同时 reset gating。
2. 增加 action trust-region。
3. 增加 accept/reject。
4. 增加 uncertainty hard threshold。
5. 增加 intervention 诊断日志。
6. 默认保守参数：

```yaml
steering:
  n_steps: 1 或 3
  step_size: 0.001 ~ 0.003
  noise_scale: 0.0 或很小
  kl_weight: 30 ~ 100
  action_delta_max: 0.03 ~ 0.08
  accept_energy_margin: 0.0 ~ 0.1

bilateral_gating:
  lambda_max: 0.1 ~ 0.3
  hard_uncertainty_threshold: 待 calibration
```

验证：

```text
FLOWER clean 50/100 chains：L1 不掉
FLOWER clean 200 chains：L1/L2/L3 不显著下降，L5 持平或提升
```

## Stage 2：Reranking-first CE-AIS

目标：减少梯度 steering 对强模型 action manifold 的破坏。

改动：

1. VLA action 周围采样 K 个候选。
2. CE-WM 对候选打分。
3. 选择最低风险且接近 VLA 的候选。
4. 只在 accepted candidate 上可选做 1-step small Langevin。

验证：

```text
clean preservation
failure-subset recovery
latency vs K sweep
```

## Stage 3：CE-WM 训练稳定化

目标：解决 margin 爆炸和 loss 坍塌。

改动：

1. energy scale regularization；
2. target margin；
3. grad norm regularization/monitoring；
4. validation AUC 和 calibration；
5. hard negatives；
6. checkpoint selection 不再只看 loss。

验证：

```text
margin 不爆炸
energy scale 稳定
grad norm 稳定
steering smoke 不伤 frozen
```

## Stage 4：Goal-conditioned / progress-aware CE-WM

目标：让 CE-AIS 真正服务任务成功，而不是只判断动作像不像数据。

改动：

1. instruction embedding 输入 CE-WM；
2. 加 progress/success head；
3. 用 failed rollout 和 stuck windows 做训练；
4. score 中加入 progress term。

验证：

```text
L5 conditional success
stuck recovery
recovery curve
```

## Stage 5：跨模型通用 CE-WM

目标：证明 CE-AIS 是通用模块，不是 FLOWER-specific hack。

改动：

1. 收集 FLOWER / RoboFlamingo / RoboVLMs rollout。
2. 训练 mixed-source CE-WM。
3. 同一个 CE-WM 接多个 VLAAdapter。

验证：

```text
CE-WM trained once, reused across multiple frozen action models
```

---

## 8. 论文叙事修订建议

### 不建议继续强调

```text
CE-AIS should greatly improve clean ABC->D on every metric.
```

原因：强基座 clean 已接近饱和，强行大幅提升不现实。

### 建议强调

```text
CE-AIS provides safe, model-agnostic test-time action verification and recovery without updating any VLA parameters.
```

核心贡献可以写成：

1. **Zero-gradient action-space TTA**：不更新 VLA，也不更新 CE-WM，只在动作空间进行安全选择。
2. **Goal-conditioned energy verifier**：用能量模型评估候选动作的风险、进度和不确定性。
3. **Abstentive active inference**：不确定或无收益时自动回退 frozen VLA，避免有害干预。
4. **Long-horizon recovery**：在长链后段、stuck 和 OOD 边界状态中提升恢复能力。
5. **Cross-model plug-and-play**：通过统一 VLAAdapter 接 FLOWER、RoboFlamingo、RoboVLMs 等动作模型。

---

## 9. 后续立即执行清单

### 代码优先级

1. 实现 Safe CE-AIS：reset gating、trust-region、accept/reject、diagnostic logging。
2. 给配置增加保守 steering 参数。
3. 给评估脚本输出 intervention diagnostics。
4. 实现 OOD severity sweep。
5. 实现 failure-subset 和 conditional success 统计。
6. 修改 CE-WM 训练 loss，加入 energy regularization、target margin 和 grad norm monitoring。
7. 后续再做 goal-conditioned/progress-aware CE-WM。

### 实验优先级

1. FLOWER clean 50-chain smoke：确认 Safe CE-AIS 不掉 L1。
2. FLOWER clean 200-chain：对比 frozen vs CE-AIS-safe。
3. FLOWER mild/medium OOD：寻找稳定提升区间。
4. Recovery 曲线：作为论文重点图。
5. Failure-subset：证明 CE-AIS 能救 frozen 失败链。
6. RoboFlamingo/RoboVLMs 接入后做 cross-model 表。

---

## 10. 最终目标

短期目标不是立刻让所有指标大涨，而是先实现：

```text
CE-AIS-safe clean 不伤强基座。
```

中期目标：

```text
mild/medium OOD 和 recovery 上稳定提升。
```

最终论文目标：

```text
同一个 CE-AIS 模块在多个 frozen action model 后面工作，clean 保持、OOD/recovery 改善、长链失败可恢复，并且全程不更新 VLA 参数。
```

这条路线比当前“每步 Langevin 强制偏转”更稳，也更符合 `/data0/yejinxuan/workspace/robot/创新二修改_backup_v1.md` 中 CE-AIS 的顶会级理论定位。
