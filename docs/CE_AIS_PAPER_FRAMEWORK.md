# CE-AIS: Causal-Energy Active Inference Steering for Test-Time Robotic Action Correction

## 1. Problem Statement

大规模视觉-语言-动作（VLA）模型在机器人操作任务中展现了强大的泛化能力，但在部署时面临严重的分布外（OOD）性能退化问题：

- **协变量偏移**：光照、纹理、相机位姿等视觉条件变化导致感知失配
- **动力学漂移**：摩擦系数、负载质量等物理参数变化导致控制失配
- **长时序误差累积**：多步任务中微小预测误差逐步放大，最终导致任务失败

现有推理时自适应方法（如 AdaWorldPolicy、TT-VLA）依赖参数微调，存在三个根本缺陷：
1. 梯度回传的计算延迟不满足实时控制要求（<50ms/step）
2. 持续参数更新引发灾难性遗忘，破坏大规模预训练积累的通用知识
3. 小幅权重扰动可能引发混沌控制轨迹

**核心问题**：能否在完全冻结所有模型参数的前提下，仅通过动作空间优化实现安全、高效的推理时自适应？

---

## 2. Method Overview

### 2.1 核心思想

CE-AIS 提出一种全新的推理时自适应范式：**零梯度动作空间偏转**。与参数微调方法相反，CE-AIS 将所有网络参数（VLA、编码器、世界模型）完全冻结，仅对 VLA 输出的动作张量本身进行信赖域约束下的能量引导优化。

### 2.2 系统架构：非对称双流推理拓扑

```
┌─────────────────────────────────────────────────────┐
│  主语义策略流 (Frozen VLA)                           │
│  FLOWER / OpenVLA / π0 / Diffusion Policy           │
│  输出: 候选动作先验 a₀                               │
└────────────────────┬────────────────────────────────┘
                     │ a₀
                     ▼
┌─────────────────────────────────────────────────────┐
│  因果能量裁判流 (Frozen CE-WM)                       │
│  ├─ Mamba-3 时序引擎 (100-300M params)              │
│  ├─ MC-Dropout 认知不确定性估计                      │
│  ├─ 双向门控强度计算 λ(u_t)                          │
│  ├─ 退火朗之万动力学 EFE 偏转                        │
│  └─ Accept/Reject 安全决策                           │
│  输出: 校正动作 a* 或 安全回退 a₀                     │
└─────────────────────────────────────────────────────┘
```

### 2.3 四大核心技术组件

#### Component 1: 因果能量世界模型（CE-WM）

- **架构**：输入投影层 → N层 Mamba-3 堆叠 → MLP 能量头
- **输入**：潜变量序列 z_{1:T} 与候选动作序列 a_{1:T}
- **输出**：标量能量值 E ∈ R，评估动作的物理合法性
- **关键设计**：判别式（非生成式），不预测未来图像，避免世界模型幻觉问题
- **训练目标**：Calibrated NCE Loss + 能量正则化 + Margin 约束

$$\mathcal{L} = \mathcal{L}_{\text{NCE}} + \lambda_E \cdot \mathbb{E}[E^2_{\text{pos}} + E^2_{\text{neg}}] + \lambda_{\uparrow} \cdot \max(0, \Delta E - \Delta_{\text{target}})^2 + \lambda_{\downarrow} \cdot \max(0, \Delta_{\min} - \Delta E)^2$$

#### Component 2: 信赖域约束退火朗之万动力学

在动作空间中执行受约束的能量下降优化：

$$a^* = \arg\min_a \left[ E_\phi(z_t, a) + \beta \|a - a_0\|^2_2 \right] \quad \text{s.t.} \quad \|a - a_0\|_\infty \leq \delta$$

- 退火策略：步长随迭代衰减，前期探索、后期收敛
- 信赖域：硬约束 δ（默认 0.08），确保偏转幅度有界
- 有限差分梯度：仅对动作张量（7-8维）计算，无需网络反向传播

#### Component 3: MC-Dropout 双向不确定性门控

$$\lambda(u_t) = \lambda_{\max} \cdot \exp\left(-\frac{(u_t - \mu_u)^2}{2\sigma_u^2}\right)$$

- 低不确定性 → λ → λ_max：信任 CE-WM，强力纠偏
- 高不确定性 → λ → 0：不信任能量梯度，回退至 VLA 保守先验
- 防止极端 OOD 场景下错误能量梯度毒化动作流

#### Component 4: Abstentive Accept/Reject 安全层

候选动作 a* 需同时满足以下条件才被执行：
1. 能量改善：E(z_t, a*) < E(z_t, a₀) - margin
2. 不确定性可控：u_t ≤ threshold
3. 信赖域满足：‖a* - a₀‖∞ ≤ δ
4. 数值安全：所有值有限

任一条件不满足则回退至原始 VLA 动作（fail-safe）。

---

## 3. Key Properties

| 属性 | CE-AIS | 参数微调方法 (AdaWorldPolicy等) |
|------|--------|-------------------------------|
| 参数更新 | 无（100% 冻结） | 需要梯度回传 |
| 灾难性遗忘 | 不存在 | 持续恶化 |
| 计算模式 | 动作空间有限差分 | 网络权重反向传播 |
| 安全保证 | 内置 Accept/Reject | 无 |
| 模型无关性 | 任意 VLA 即插即用 | 需适配特定架构 |
| 可解释性 | 能量/不确定性/门控完全可观测 | 黑盒 |

---

## 4. Experiments

### 4.1 评估基准

- **CALVIN**（ABC→D）：长时序多步操作任务，5步链式评估
- **LIBERO**（Spatial/Object/Goal/Long）：多场景泛化评估

### 4.2 基线模型（作为 CE-AIS 的 VLA 载体）

- FLOWER（扩散策略，语言条件）
- OpenVLA-OFT
- Diffusion Policy
- BC-Transformer

### 4.3 实验设计

1. **Clean 性能保持**：验证 CE-AIS 不损害已优基线性能
2. **OOD 鲁棒性**：视觉/物理/相机扰动下的成功率对比
3. **消融实验**：信赖域 / Accept-Reject / 不确定性门控 / CE-WM 损失变体
4. **长时序恢复**：多步任务中间失败后的纠错能力
5. **计算效率**：延迟/FLOPS Pareto 曲线

### 4.4 初步结果（CALVIN ABC→D, FLOWER 基线）

| 指标 | Frozen FLOWER | + CE-AIS (uncertainty_gated) |
|------|---------------|------------------------------|
| L1   | 99.8%         | 99.8% (±0%)                  |
| L2   | 96.8%         | 96.9% (+0.1%)                |
| L3   | 91.9%         | 92.1% (+0.2%)                |
| L4   | 86.1%         | 86.5% (+0.4%)                |
| L5   | 78.8%         | 79.2% (+0.4%)                |
| Avg  | 4.534         | 4.545 (+0.011)               |

特点：完全保持基线性能的同时，在长时序步骤（L4/L5）上取得稳定提升。

---

## 5. Contributions

1. **新范式**：首次提出完全冻结参数的推理时动作空间偏转框架，从根本上消除灾难性遗忘
2. **因果能量世界模型**：判别式（非生成式）的状态-动作能量评估，避免世界模型幻觉
3. **理论连接**：将认知科学中的主动推理（Active Inference）与能量模型桥接到具身AI控制
4. **工程价值**：模型无关的即插即用安全层，内置多层安全机制，具备实际部署可行性

---

## 6. Paper Structure (Planned)

1. Introduction & Motivation
2. Related Work (TTA for Embodied AI, Energy-Based Models, Active Inference)
3. Method: CE-AIS Framework
   - 3.1 Problem Formulation
   - 3.2 Causal Energy World Model
   - 3.3 Trust-Region Langevin Steering
   - 3.4 Bilateral Uncertainty Gating
   - 3.5 Accept/Reject Safety Layer
4. Experiments
   - 4.1 CALVIN Long-Horizon Evaluation
   - 4.2 LIBERO Multi-Model Validation
   - 4.3 OOD Robustness Analysis
   - 4.4 Ablation Studies
   - 4.5 Computational Efficiency
5. Analysis & Discussion
6. Conclusion
