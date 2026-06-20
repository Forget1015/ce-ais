# CE-WM 重训计划：修复 Energy Landscape 结构性缺陷

## 1. 问题诊断

### 1.1 现象

CE-AIS v2 Gated 全量 1000 chains：
- 83 条提升，80 条退步，837 条持平
- 净增仅 11 个 tasks（Avg Len +0.011）
- 提升和退步几乎对半 → steering 效果接近随机扰动

### 1.2 训练流程回顾

**Stage 1: Base Pretrain (cewm_epoch0033.pt)**
- 数据：ABC_D 原始数据集（图片 → encoder → z_seq）
- 正样本：专家 action（人类 teleoperation）
- 负样本：PerturbationRegistry（velocity_reversal, random_displacement 等大尺度扰动）
- Loss：NCE（E(pos) < E(neg)）
- 33 epochs

**Stage 2: Energy Head Finetune (energy_head_v2_best.pt)**
- 数据：contrastive_pairs.npz（FAISS 近邻配对的成功/失败 action）
- 冻结 backbone，只训 energy head
- 20 epochs，acc=0.890

### 1.3 根因：Energy Landscape 结构性缺陷

**关键实验：从 expert action 出发，沿不同方向走，观察 energy 变化：**

| 方向 | alpha=0 | alpha=0.1 | alpha=0.5 | alpha=1.0 |
|------|---------|-----------|-----------|-----------|
| pos→neg（走向失败） | -1.47 | -0.93 | +0.53 | **+1.34** |
| pos→random（随机方向） | -1.47 | -1.86 | -5.25 | **-7.34** |

**发现：expert action 不是 energy 的全局最低点！**

- 往失败方向走：energy 单调上升 ✓（正确）
- 往随机方向走：energy 大幅下降 ✗（错误）

**这意味着 CE-AIS 的 steering（沿 -gradient 降低 energy）实际上在把 action 从 expert 区域拉走！**

### 1.4 为什么会这样

NCE/ranking loss 只约束了 `E(pos) < E(neg)`，没有约束 `E(pos) < E(任何其他 action)`。Model 找到了 shortcut：把特定的 neg 的 energy 推高，但 action space 中大片区域的 energy 都低于 expert。

### 1.5 为什么 83 条提升 / 80 条退步

CE-AIS 的 0.0003 correction 相对于成功/失败差距（1.13）极小。效果本质上是随机扰动在机器人控制分岔点（如抓取瞬间）的运气，不是 model 有意义的引导。MC-dropout 随机性实验也证实了这一点（同配置跑两次 Avg 差 0.7）。

---

## 2. 解决方案

### 2.1 核心思路

让 expert action 成为 energy landscape 的**全局最低点**（或至少是局部盆地底部），使得任何方向偏离 expert 都导致 energy 上升。这样 steering 沿 -gradient 走才会正确地走向 expert。

### 2.2 参考文献中的成熟方法

1. **Implicit Behavioral Cloning (Florence et al. 2022)**：在 action space 中均匀采样大量负样本，用 InfoNCE loss。均匀采样覆盖整个 action space，自然约束 energy 只在 expert 处最低。

2. **Contrastive Divergence (Hinton 2002)**：用 MCMC 从 model 当前 low-energy 区域采样负样本。直接修复"model 认为某些非 expert 区域 energy 低"的问题。

3. **Energy Regularization**：对非 expert 区域加正则 `L_reg = max(0, threshold - E(random))`，强制随机 action energy 不能低于阈值。

### 2.3 具体方案：多源负样本 + Ranking Loss

在 base pretrain 阶段改造负样本生成，混合四类负样本：

| 类型 | 做法 | 目的 | 占比 |
|------|------|------|------|
| 均匀采样 | 从 [-1,1]^7 均匀采 action | 覆盖整个 action space，堵住 energy 漏洞 | 30% |
| 多尺度高斯 | expert + N(0, σ), σ∈{0.01, 0.05, 0.1, 0.3} | 确保 expert 附近各尺度 gradient 正确 | 30% |
| 原始 PerturbationRegistry | velocity_reversal, collision 等 | 识别极端物理违规 | 20% |
| MCMC hard negative（可选） | 从当前 model low-energy 区域采样 | 精准修复当前 landscape 漏洞 | 20% |

初期不加 MCMC（实现复杂），最后 20% 用均匀采样替代。

### 2.4 Loss 设计

```
L = L_ranking + α × L_nce + β × L_reg

L_ranking: E(expert) < E(gauss_small) < E(gauss_large) < E(uniform)
  margin 按尺度递增: 0.2, 0.5, 1.0

L_nce: 原始 NCE loss（expert vs PerturbationRegistry negatives）
  α = 0.5

L_reg: energy regularization
  对均匀采样的 action: max(0, -1.0 - E(uniform))
  确保均匀采样 action 的 energy 不低于 -1.0
  β = 0.1
```

### 2.5 配套 Steering 配置调整

重训后 landscape 正确，同步增大 correction 幅度：

```yaml
steering:
  n_steps: 3
  step_size: 0.008
  kl_weight: 10.0
  action_delta_max: 0.08
  enable_accept_reject: true
  accept_energy_margin: 0.0
```

预期 correction 幅度：~0.01-0.05（进入 model 训练过的尺度）。

---

## 3. 实现计划

### 3.1 修改 PerturbationRegistry

在 `src/data/perturbation.py` 中新增：
- `uniform_sampling`：从 [-1,1]^7 均匀采样
- `multi_scale_gaussian`：从多个 σ 中随机选一个加噪声
- 已有的 `micro_perturbation`, `medium_perturbation`, `large_perturbation` 可复用

### 3.2 修改 pretrain_pipeline.py

- 改造 `pretrain_cewm()` 中的负样本生成逻辑
- 混合多源负样本（均匀 + 高斯 + 原始策略）
- 新增 ranking loss 和 energy regularization
- 保持原有训练框架（accelerator, tqdm, checkpoint 等）

### 3.3 训练配置

| 参数 | 值 |
|------|-----|
| 数据 | ABC_D training split（和原 pretrain 相同） |
| epochs | 40 |
| batch_size | 128（图片数据，受 GPU 显存限制） |
| lr | 1e-4 |
| K (neg/pos ratio) | 5 |
| 负样本组成 | 2 uniform + 2 multi_scale_gauss + 1 orig_perturbation |
| GPU | 单卡 A100 80G |
| 预计时间 | 和原 pretrain 类似（需要读图片 + encoder forward） |

### 3.4 验证流程

1. 训练中监控：每 5 epochs 验证 energy landscape 是否修复
   - E(expert) < E(uniform)? 比例应 >90%
   - E(expert) < E(gauss_0.1)? 比例应 >85%
   - 从 expert 出发随机方向 energy 是否上升?

2. 训练完成后：
   - 40-chain fast test × 3 次取平均
   - 全量 1000 chains 对比

---

## 4. 风险与备选

| 风险 | 应对 |
|------|------|
| 均匀负样本太简单（action space 大部分是"明显错误"） | 加 hard negative mining（MCMC 采样） |
| Multimodal expert action（同一 state 多个合法 action） | Ranking loss 不要求绝对最低，只要求 expert 比 random 低 |
| 训练时间过长（图片数据 IO 瓶颈） | 使用 mmap 后端（已支持），或预计算 z_seq 缓存 |
| 重训后 correction 仍太小 | 配合 n_steps=3 增大步数 |

---

## 5. 时间线

- [x] 修改 perturbation.py 新增 uniform_sampling
- [x] 修改 pretrain_pipeline.py 改造负样本 + loss
- [x] 创建新的训练配置
- [x] 启动训练（4卡并行，~1.6h/epoch）
- [ ] 验证 energy landscape + fast test
- [ ] 全量 1000 chains 对比

---

## 6. 进一步分析（2026-06-18 更新）

### 6.1 旧版 base pretrain (epoch33) 的 landscape 并不太差

对旧版微调前的 `cewm_epoch0033.pt` 做了相同的 landscape 测试：

| 测试 | 旧 base pretrain (epoch33) | 旧 finetune 后 (v2_best) | 新 pretrain (epoch9) |
|------|---|---|---|
| 区分成功/失败 action | 49.8%（不行） | **94.0%** ✓ | 55.4%（不行） |
| σ=0.1 偏移 energy 上升 | 60.0% | 很低 | 76.7% |
| σ=0.5 偏移 energy 上升 | **80.2%** | 4%（反转） | 96.3% |
| Uniform energy 上升 | **85.2%** | 很低 | 91.7% |

**关键发现：旧版 base pretrain 的 landscape 本来就还行（80-85%），是 finetune 阶段把 landscape 搞坏的。**

finetune 只训了 energy head，为了让 model 区分成功/失败 action（94%），energy head 学到了一种 shortcut：扭曲 landscape 形状，把特定 action 的 energy 拉低/推高，代价是整体 landscape 结构被破坏（随机方向 energy 下降到 96%）。

### 6.2 多尺度预训练在小尺度上改善有限

| σ 尺度 | 旧 base (epoch33) | 新 pretrain (epoch9) | 改善 |
|--------|---|---|---|
| 0.01 | 50.8% | 51.7% | +0.9%（无效） |
| 0.05 | 60.2% | 60.0% | -0.2%（无效） |
| 0.10 | 60.0% | 76.7% | +16.7% |
| 0.30 | 73.0% | 93.0% | +20.0% |
| 0.50 | 80.2% | 96.3% | +16.1% |

**结论：多尺度训练在 σ≥0.1 有明显改善，但 σ≤0.05 完全没效果。**

原因是架构层面的限制：`input_proj: Linear(135→640)`，action 只占 7/135=5% 的输入。σ=0.01 的 action 扰动经过 input_proj 后变化量约 0.01%，过 32 层 Mamba 后信号消失。这不是训练数据能解决的。

### 6.3 Gradient-Based Steering 的天花板

即使 landscape 完美，CE-AIS 用 gradient 做修正面临的根本问题：

1. **小尺度 gradient 不可靠**：model 在 σ=0.01-0.05（CE-AIS n_steps=3 的工作尺度）上方向准确率只有 51-76%
2. **修正幅度 vs 真实差距**：成功/失败 action 平均差距 L_inf=1.13，即使 correction 增大到 0.05 也只是真实差距的 4.4%
3. **累积效果不确定**：76% 正确率 × 每步 0.05 幅度 × 100 步，净效果可能正向但很弱

**gradient-based steering 在当前架构下的天花板可能很低。**

### 6.4 Rerank 方案的可行性分析

**FLOWER 成功/失败 action 的实际差距：**

| 维度 | 平均差距 | 最大差距 |
|------|---------|---------|
| dx | 0.2475 | 1.68 |
| dy | 0.2088 | 1.63 |
| dz | 0.1987 | 1.25 |
| gripper | **0.8831** | 2.00 |

- L_inf 中位数 = 0.84，p25 = 0.38
- gripper 维度贡献最大差距

**候选覆盖率（从 pos 出发 ± 2σ 能覆盖到 neg 的比例）：**

| σ | 覆盖率 |
|---|--------|
| 0.05 | 1.1% |
| 0.10 | 7.5% |
| 0.20 | 26.8% |
| 0.30 | 41.0% |
| 0.50 | 53.3% |

**分析：** σ=0.3-0.5 只能覆盖 41-53% 的失败情况。但 rerank 的目的不是"从失败跳到成功"，而是在 FLOWER 已经大部分正确的前提下，在关键时刻选一个稍好的方向。

**Rerank 的优势：** model 在大尺度（σ=0.3-0.5）上排序正确率 80-93%，做"N 选 1"的选择题比"判断梯度方向"可靠得多。

### 6.5 相关工作与可能方向

| 方向 | 方法 | 优势 | 和 CE-AIS 的关系 |
|------|------|------|----------------|
| Test-Time Gradient Guidance | 用 value/energy function 的梯度引导 policy 输出 | 和 CE-AIS 思路一致 | 面临同样的小尺度 gradient 不可靠问题 |
| CEM/Rerank | 采样 N 个候选，用 model 评分选最好的 | 不依赖 gradient 方向，只需要排序 | 可替换 CE-AIS 的 steering 模块 |
| MPC (TD-MPC) | world model 多步 rollout + CEM 优化 | 考虑未来多步效果 | 计算量大但更 principled |
| Geometric Action Model | 结构化 action representation | action 信息不被淹没 | 可改进 CE-WM 的 action 表示 |
| Action Embedding Pathway | action 独立 encoder + cross-attention | 增大 action 在模型中的影响力 | 架构改动，需要重训 |

### 6.6 当前的选择

**核心判断：gradient-based steering 在当前架构下天花板有限。**

两个改进方向（不改变 CE-AIS "因果能量世界模型做 test-time correction" 的核心创新）：

**方向 A：Steering 从 gradient 改为 rerank（不改模型，改推理方式）**
- 保持 CE-WM 不变
- 从 FLOWER 输出附近生成 N 个候选
- 用 CE-WM energy 评分，选最低的执行
- 优势：不需要重训，model 大尺度排序能力已有 80-93%
- 风险：候选覆盖范围 vs 修正幅度的平衡

**方向 B：改 action 在模型中的表示（改架构，需要重训）**
- 给 action 独立的 embedding pathway（7 维 → 64/128 维）
- 让 action 微小变化在模型中有更大的表示力
- 优势：从根本上解决"小尺度信号被淹没"的问题
- 风险：需要重训，改动较大

**推荐顺序：先试 A，再考虑 B。** A 不需要重训，几小时内就能实现和验证。如果 A 有效，证明 CE-WM 本身是好的，只是 gradient steering 不是最佳使用方式。论文叙事也更好："CE-WM 作为 energy scorer + sampling-based optimization"。
