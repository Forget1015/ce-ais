# CE-AIS 方法论公式化说明

本文档将当前 CE-AIS 代码实现整理为论文式方法描述，重点说明完整输入输出流程、训练目标、测试时推理算法、能量模型、主动推理式动作偏转、安全信任域、接受/拒绝机制以及诊断指标。本文以当前代码中的 Safe CE-AIS 实现为准，将其抽象为一个冻结 VLA 上的测试时动作空间控制框架。

---

## 1. 方法总览

CE-AIS（Causal-Energy Active Inference Steering）的核心目标是在不更新 VLA 参数的前提下，为已有视觉语言动作模型提供一个可插拔的测试时安全校正层。给定机器人当前观测和语言指令，基础 VLA 先生成动作先验，随后冻结的对比编码器将观测映射到潜在状态，冻结的因果能量世界模型对动作序列进行能量评估和不确定性估计。CE-AIS 根据不确定性门控决定是否干预，并在动作空间内执行受信任域约束的主动推理式能量下降。最后，系统只在校正动作满足能量改进、安全信任域和数值稳定性条件时接受校正，否则回退到原始 VLA 动作。

整体上，CE-AIS 可写为：

$$
\begin{aligned}
\mathbf{a}_0 &= \pi_{\mathrm{VLA}}(\mathbf{o}_t, g), \\
\mathbf{z}_t &= f_\phi(\mathbf{o}_t), \\
E_0 &= E_\theta(\mathbf{z}_{1:T}, \mathbf{a}_0), \\
u_t &= \mathrm{Unc}_\theta(\mathbf{z}_{1:T}, \mathbf{a}_0), \\
\lambda_t &= \Gamma(u_t; \mathcal{H}_u), \\
\tilde{\mathbf{a}} &= \mathcal{S}(\mathbf{a}_0, \mathbf{z}_t, E_\theta, \lambda_t), \\
\mathbf{a}^{\star} &= \Pi_{\Delta}(\tilde{\mathbf{a}}; \mathbf{a}_0), \\
\mathbf{a}_{\mathrm{exec}} &= \mathcal{A}(\mathbf{a}^{\star}, \mathbf{a}_0, E_\theta, u_t),
\end{aligned}
$$

其中：

- $\pi_{\mathrm{VLA}}$ 是冻结的基础 VLA 策略；
- $f_\phi$ 是冻结或预训练后的对比编码器；
- $E_\theta$ 是冻结的因果能量世界模型 CE-WM；
- $\Gamma$ 是双向不确定性门控；
- $\mathcal{S}$ 是 EFE / Langevin 动作偏转算子；
- $\Pi_\Delta$ 是相对 VLA 先验的信任域投影；
- $\mathcal{A}$ 是安全接受/拒绝规则。

该设计的关键约束是：测试时不更新任何模型参数，只在动作空间上进行有限步、局部、可拒绝的校正。

---

## 2. 符号定义

| 符号 | 含义 |
|---|---|
| $t$ | 当前环境交互步 |
| $T$ | VLA 输出动作 chunk 长度 |
| $d_a$ | 单步动作维度，当前 CALVIN/FLOWER 设置为 7 |
| $d_z$ | 潜在状态维度，当前配置为 128 |
| $\mathbf{o}_t$ | 当前观测 |
| $g$ | 语言目标或语言指令 |
| $\mathbf{a}_0 \in \mathbb{R}^{B\times T\times d_a}$ | VLA 输出的初始动作序列，也称动作先验 |
| $\mathbf{z}_t \in \mathbb{R}^{B\times d_z}$ | 编码器输出的当前潜在状态 |
| $\mathbf{z}_{1:T}$ | 为匹配动作 chunk 而展开的潜在状态序列 |
| $E_\theta(\mathbf{z}_{1:T},\mathbf{a}_{1:T})$ | CE-WM 输出的能量，越低表示越接近专家/合法动作分布 |
| $u_t$ | MC-Dropout 估计的不确定性 |
| $\lambda_t$ | 不确定性门控强度，控制 CE-AIS 干预幅度 |
| $\delta_{\max}$ | 信任域最大逐元素动作偏移 |
| $\kappa$ | KL/先验保持项权重，代码中为 `kl_weight` |
| $\epsilon_k$ | 第 $k$ 步 Langevin 步长 |
| $m_{\mathrm{acc}}$ | 接受校正所需的最小能量改进 margin |

在 CALVIN/FLOWER 实验中，动作为 7 维：

$$
\mathbf{a}_t = [\Delta x, \Delta y, \Delta z, r_1, r_2, r_3, g_r] \in \mathbb{R}^7,
$$

其中前三维通常对应末端执行器平移，后三维对应旋转，最后一维对应夹爪控制。

---

## 3. 输入输出定义

### 3.1 单步输入

CE-AIS 在测试时每一步接收：

$$
(\mathbf{o}_t, g),
$$

其中观测可抽象为：

$$
\mathbf{o}_t = (I_t^{\mathrm{rgb}}, D_t, \mathbf{p}_t, \mathbf{o}_t^{\mathrm{raw}}),
$$

包含：

- RGB 图像 $I_t^{\mathrm{rgb}}$，如 CALVIN 的 static camera 和 gripper camera；
- 深度图 $D_t$；
- 机器人状态或位姿 $\mathbf{p}_t$，例如 `robot_obs[:7]`；
- 原始 CALVIN 观测 $\mathbf{o}_t^{\mathrm{raw}}$，供 FLOWER adapter 构造模型输入。

语言目标为：

$$
g \in \mathcal{G},
$$

例如 `"open the drawer"`、`"move the slider left"` 等 CALVIN 子任务语言描述。

### 3.2 单步输出

CE-AIS 输出：

$$
(\mathbf{a}_{\mathrm{exec}}, \mathcal{I}_t),
$$

其中：

- $\mathbf{a}_{\mathrm{exec}} \in \mathbb{R}^{B\times T\times d_a}$ 是最终执行动作 chunk；
- $\mathcal{I}_t$ 是诊断信息，包括状态、是否接受、是否干预、能量前后值、不确定性、门控强度、动作偏移等。

诊断状态取值为：

$$
\mathrm{status}_t \in \{\mathrm{accepted}, \mathrm{rejected}, \mathrm{abstained\_uncertainty}, \mathrm{vla\_prior}, \mathrm{fallback}\}.
$$

---

## 4. VLA 动作先验

基础 VLA 通过统一 adapter 接口表示为冻结策略：

$$
\pi_{\mathrm{VLA}}: (\mathbf{o}_t, g) \mapsto \mathbf{a}_0.
$$

代码中的调用为：

$$
\mathbf{a}_0 = \mathrm{predict}_{\mathrm{VLA}}(\mathbf{o}_t, g).
$$

对于 FLOWER，adapter 将 CALVIN 原始观测中的 static/gripper RGB 图像转换为模型输入，执行：

$$
\mathbf{a}_0 = \pi_{\mathrm{FLOWER}}(\mathbf{o}_t^{\mathrm{FLOWER}}, g),
$$

其中图像先被归一化为：

$$
\hat{I} = \frac{\mathrm{Resize}(I)/255 - \mu}{\sigma}.
$$

若 FLOWER 输出 chunk 长度与 CE-AIS 期望长度不一致，则 adapter 截断或重复最后一个动作，使其满足：

$$
\mathbf{a}_0 \in \mathbb{R}^{B\times T\times d_a}.
$$

CE-AIS 将 $\mathbf{a}_0$ 视为强先验，而不是可任意覆盖的初始点。因此后续所有校正都围绕 $\mathbf{a}_0$ 做局部搜索。

---

## 5. 对比编码器

### 5.1 编码器映射

对比编码器将视觉、深度和机器人位姿编码到单位球面上的潜在状态：

$$
\mathbf{z}_t = f_\phi(I_t^{\mathrm{rgb}}, D_t, \mathbf{p}_t) \in \mathbb{R}^{d_z}.
$$

代码实现可抽象为三路特征提取：

$$
\begin{aligned}
\mathbf{v}^{\mathrm{rgb}}_t &= b_{\mathrm{rgb}}(I_t^{\mathrm{rgb}}), \\
\mathbf{v}^{\mathrm{depth}}_t &= b_{\mathrm{depth}}(D_t), \\
\mathbf{v}^{\mathrm{pose}}_t &= b_{\mathrm{pose}}(\mathbf{p}_t), \\
\mathbf{h}_t &= [\mathbf{v}^{\mathrm{rgb}}_t; \mathbf{v}^{\mathrm{depth}}_t; \mathbf{v}^{\mathrm{pose}}_t], \\
\tilde{\mathbf{z}}_t &= q_\phi(\mathbf{h}_t), \\
\mathbf{z}_t &= \frac{\tilde{\mathbf{z}}_t}{\|\tilde{\mathbf{z}}_t\|_2}.
\end{aligned}
$$

其中 $[\cdot;\cdot]$ 表示特征拼接。当前实现中，RGB 和 depth 经过视觉 backbone，pose 经过 MLP，最后由 fusion head 生成 $d_z=128$ 维潜在表示。

### 5.2 序列展开

CE-WM 接收状态-动作序列。测试时只有当前状态 $\mathbf{z}_t$，因此代码将其沿 chunk 维展开：

$$
\mathbf{z}_{1:T} = [\mathbf{z}_t, \mathbf{z}_t, \ldots, \mathbf{z}_t] \in \mathbb{R}^{B\times T\times d_z}.
$$

即：

$$
\mathbf{z}_{\tau} = \mathbf{z}_t, \quad \tau=1,\ldots,T.
$$

---

## 6. 编码器预训练目标：InfoNCE

编码器通过时序邻近样本对进行对比学习。给定 batch 内 anchor 表示 $\mathbf{z}_i$ 和正样本表示 $\mathbf{z}_i^+$，先进行 L2 归一化：

$$
\bar{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2},
\quad
\bar{\mathbf{z}}_i^+ = \frac{\mathbf{z}_i^+}{\|\mathbf{z}_i^+\|_2}.
$$

相似度定义为余弦相似度：

$$
s_{ij} = \bar{\mathbf{z}}_i^\top \bar{\mathbf{z}}_j^+.
$$

InfoNCE 损失为：

$$
\mathcal{L}_{\mathrm{InfoNCE}}
= -\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(s_{ii}/\tau_e)}
{\sum_{j=1}^{B}\exp(s_{ij}/\tau_e)},
$$

其中 $\tau_e$ 是 encoder temperature，当前配置中为：

$$
\tau_e = 0.07.
$$

代码中使用 batch 内其他样本作为负样本，对应分类标签：

$$
y_i = i.
$$

因此 InfoNCE 等价于对 logits 矩阵：

$$
\mathbf{L}_{ij} = \frac{\bar{\mathbf{z}}_i^\top \bar{\mathbf{z}}_j^+}{\tau_e}
$$

执行交叉熵分类。

---

## 7. 因果能量世界模型 CE-WM

### 7.1 能量函数定义

CE-WM 学习一个状态-动作序列能量函数：

$$
E_\theta: (\mathbf{z}_{1:T}, \mathbf{a}_{1:T}) \mapsto \mathbb{R}.
$$

能量越低表示该动作序列越接近专家轨迹或合法动力学分布，能量越高表示动作更可能是不合理、扰动或越界动作。

### 7.2 网络结构公式

对每个时间步，将 latent state 和 action 拼接：

$$
\mathbf{x}_\tau^{(0)} = W_{\mathrm{in}}[\mathbf{z}_\tau; \mathbf{a}_\tau] + \mathbf{b}_{\mathrm{in}},
\quad \tau=1,\ldots,T.
$$

将序列输入 Mamba/SSM 时序层：

$$
\mathbf{x}_{1:T}^{(\ell)} = \mathcal{M}_\ell(\mathbf{x}_{1:T}^{(\ell-1)}),
\quad \ell=1,\ldots,L.
$$

取最后一个时间步 hidden state：

$$
\mathbf{h}_T = \mathbf{x}_T^{(L)}.
$$

能量头输出标量能量：

$$
E_\theta(\mathbf{z}_{1:T}, \mathbf{a}_{1:T}) = h_E(\mathbf{h}_T) \in \mathbb{R}.
$$

当前 `base.yaml` 中 CE-WM 主要参数为：

$$
\begin{aligned}
d_z &= 128, \\
d_a &= 7, \\
d_{\mathrm{model}} &= 640, \\
L &= 32, \\
d_{\mathrm{state}} &= 64, \\
\mathrm{expand\_factor} &= 3, \\
\mathrm{mimo\_groups} &= 4, \\
\mathrm{dropout} &= 0.1.
\end{aligned}
$$

---

## 8. CE-WM 训练目标：校准 NCE

### 8.1 正负样本构造

CE-WM 训练样本由专家正样本和扰动负样本组成：

$$
\mathcal{D}_{\mathrm{CEWM}} = \{(\mathbf{z}_{1:T}^{(i)}, \mathbf{a}_{+,1:T}^{(i)}, \{\mathbf{a}_{-,k,1:T}^{(i)}\}_{k=1}^{K})\}_{i=1}^{N}.
$$

其中：

- $\mathbf{a}_+$ 是数据集中真实专家动作；
- $\mathbf{a}_{-,k}$ 是对专家动作施加扰动得到的负样本；
- $K$ 是负样本数，当前配置中 `neg_sample_ratio=5`。

若 batch 中包含原始图像序列，则训练时在线编码：

$$
\mathbf{z}_{1:T} = f_\phi(\mathbf{o}_{1:T}).
$$

并冻结 encoder 参数，即：

$$
\nabla_\phi \mathcal{L}_{\mathrm{CEWM}} = 0.
$$

### 8.2 负样本扰动族

代码中的负样本扰动可抽象为一组非法或次优动作变换：

$$
\mathcal{P} = \{P_1, P_2, \ldots, P_M\}.
$$

对正样本动作应用扰动：

$$
\mathbf{a}_{-,k} = P_{k \bmod M}(\mathbf{a}_+).
$$

主要扰动包括：

1. 速度反转：

$$
P_{\mathrm{rev}}(\mathbf{a})_{1:3} = -\mathbf{a}_{1:3}.
$$

2. 夹爪异常：

$$
P_{\mathrm{grip}}(\mathbf{a})_{g} \sim \mathrm{RandomGrip}.
$$

3. 随机位移：

$$
P_{\mathrm{disp}}(\mathbf{a}) = \mathbf{a} + \boldsymbol{\epsilon},
\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_p^2 I).
$$

4. 碰撞/动力学违规：

$$
P_{\mathrm{collide}}(\mathbf{a})_{1:3} = c \cdot \mathbf{r},
$$

其中 $c$ 是放大系数，$\mathbf{r}$ 是随机方向。

5. 时间打乱：

$$
P_{\mathrm{shuffle}}(\mathbf{a}_{1:T}) = \mathbf{a}_{\pi(1:T)},
$$

其中 $\pi$ 是时间维随机置换。

6. 关节/动作边界违规：

$$
P_{\mathrm{limit}}(\mathbf{a})_d \in \{a_{\min}, a_{\max}\}.
$$

这些扰动使 CE-WM 学到专家动作与不合理动作之间的能量差。

### 8.3 NCE 能量分类损失

对一个 batch，正样本能量为：

$$
E_i^+ = E_\theta(\mathbf{z}_{1:T}^{(i)}, \mathbf{a}_{+,1:T}^{(i)}),
$$

负样本能量为：

$$
E_{i,k}^- = E_\theta(\mathbf{z}_{1:T}^{(i)}, \mathbf{a}_{-,k,1:T}^{(i)}).
$$

CE-WM 希望正样本能量低、负样本能量高。因此使用负能量作为分类 logits：

$$
\mathbf{l}_i = \frac{1}{\tau_w}
[-E_i^+, -E_{i,1}^-, \ldots, -E_{i,K}^-].
$$

正样本位于第 0 类，因此标签为：

$$
y_i = 0.
$$

NCE 损失为：

$$
\mathcal{L}_{\mathrm{NCE}}
= -\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(-E_i^+/\tau_w)}
{\exp(-E_i^+/\tau_w)+\sum_{k=1}^{K}\exp(-E_{i,k}^-/\tau_w)}.
$$

当前代码中 $\tau_w=1.0$。

### 8.4 能量间隔与校准正则

原始 NCE 容易出现能量尺度爆炸或塌缩，因此当前实现加入校准项。定义平均能量间隔：

$$
\Delta_E = \mathbb{E}_{i,k}[E_{i,k}^-] - \mathbb{E}_i[E_i^+].
$$

能量幅值正则为：

$$
\mathcal{L}_{\mathrm{reg}}
= \mathbb{E}_i[(E_i^+)^2] + \mathbb{E}_{i,k}[(E_{i,k}^-)^2].
$$

上界 margin 惩罚防止能量间隔无限变大：

$$
\mathcal{L}_{\mathrm{upper}}
= \mathrm{ReLU}(\Delta_E - m_{\mathrm{target}})^2.
$$

下界 margin 惩罚保证正负样本仍有最小分离：

$$
\mathcal{L}_{\mathrm{lower}}
= \mathrm{ReLU}(m_{\min} - \Delta_E)^2.
$$

最终 CE-WM 训练损失为：

$$
\mathcal{L}_{\mathrm{CEWM}}
= \mathcal{L}_{\mathrm{NCE}}
+ \beta_E \mathcal{L}_{\mathrm{reg}}
+ \beta_U \mathcal{L}_{\mathrm{upper}}
+ \beta_L \mathcal{L}_{\mathrm{lower}}.
$$

当前配置中：

$$
\begin{aligned}
\beta_E &= 10^{-4}, \\
m_{\mathrm{target}} &= 5.0, \\
m_{\min} &= 1.0, \\
\beta_U &= 10^{-2}, \\
\beta_L &= 1.0.
\end{aligned}
$$

该损失的目标不是让 CE-WM 分类损失无限下降，而是让其保持可用于动作梯度的稳定能量景观。经验上，若 margin 过大，能量梯度可能不再适合局部控制；若 loss 接近 $\log(1+K)$，则说明能量模型趋近随机分类。

---

## 9. 测试时不确定性估计

CE-AIS 使用 MC-Dropout 估计 CE-WM 对当前 VLA 动作先验的不确定性。令 $E_\theta^{(m)}$ 表示第 $m$ 次 dropout mask 下的能量模型输出，则：

$$
E^{(m)} = E_\theta^{(m)}(\mathbf{z}_{1:T}, \mathbf{a}_0),
\quad m=1,\ldots,M.
$$

不确定性定义为能量方差：

$$
u_t = \mathrm{Var}_{m=1}^{M}\left[E_\theta^{(m)}(\mathbf{z}_{1:T}, \mathbf{a}_0)\right].
$$

代码中对应：

$$
M = \mathrm{mc\_samples}.
$$

当前 `safe_balanced.yaml` 中：

$$
M=5.
$$

若设置了硬不确定性阈值 $u_{\max}$，则当：

$$
u_t > u_{\max}
$$

时系统直接放弃干预，输出 VLA 原始动作：

$$
\mathbf{a}_{\mathrm{exec}} = \mathbf{a}_0,
$$

并将状态记为：

$$
\mathrm{status}_t=\mathrm{abstained\_uncertainty}.
$$

当前配置中该硬阈值默认为 `null`，即不启用硬拒绝。

---

## 10. 双向不确定性门控

### 10.1 历史不确定性统计

CE-AIS 维护一个长度为 $W$ 的不确定性历史窗口：

$$
\mathcal{H}_u^t = \{u_{t-W+1}, \ldots, u_t\}.
$$

历史均值为：

$$
\mu_u^t = \frac{1}{|\mathcal{H}_u^t|}\sum_{u\in\mathcal{H}_u^t} u.
$$

代码中每次调用门控前，将当前 uncertainty 的均值加入历史队列。

### 10.2 Gaussian 门控函数

门控强度定义为：

$$
\lambda_t
= \lambda_{\max}\exp\left(
-\frac{(u_t-\mu_u^t)^2}{2\sigma_u^2}
\right),
$$

其中：

- $\lambda_{\max}$ 是最大干预强度；
- $\sigma_u$ 对应代码中的 `sensitivity`；
- $\mu_u^t$ 是当前历史窗口内的平均不确定性。

该门控是“双向”的：当不确定性显著高于历史均值或显著低于历史均值时，干预强度都会减小；只有当不确定性处于模型熟悉的稳定区间附近时，CE-AIS 才更愿意介入。

当前 `safe_balanced.yaml` 中：

$$
\lambda_{\max}=0.4, \quad \sigma_u=0.1, \quad W=50.
$$

若门控强度过低，即：

$$
\max(\lambda_t) \leq \lambda_{\min},
$$

则系统直接采用 VLA 先验：

$$
\mathbf{a}_{\mathrm{exec}}=\mathbf{a}_0,
$$

状态记为：

$$
\mathrm{status}_t=\mathrm{vla\_prior}.
$$

当前配置中：

$$
\lambda_{\min}=0.
$$

---

## 11. EFE / Langevin 动作空间偏转

### 11.1 优化目标

CE-AIS 的动作偏转可理解为在 VLA 动作先验附近求解一个局部能量最小化问题：

$$
\min_{\mathbf{a}}
\quad
E_\theta(\mathbf{z}_{1:T}, \mathbf{a})
+ \frac{\kappa}{2}\|\mathbf{a}-\mathbf{a}_0\|_2^2.
$$

其中第二项约束动作不要偏离 VLA 先验太远。该项在代码中称为 KL/proximity 项：

$$
\nabla_{\mathbf{a}} \mathcal{R}(\mathbf{a},\mathbf{a}_0)
= \kappa(\mathbf{a}-\mathbf{a}_0).
$$

在 Safe CE-AIS 中，该优化不是全局求解，而是执行有限步、带门控强度的局部下降。

### 11.2 退火步长

第 $k$ 步的步长为：

$$
\epsilon_k = \epsilon_0 \alpha^k,
$$

其中：

- $\epsilon_0$ 对应 `step_size`；
- $\alpha$ 对应 `anneal_rate`。

当前 `safe_balanced.yaml` 中：

$$
\epsilon_0=0.006, \quad \alpha=0.5, \quad K_{\mathrm{steps}}=2.
$$

### 11.3 能量梯度

若使用 autograd，则能量梯度为：

$$
\mathbf{g}_E^{(k)} = \nabla_{\mathbf{a}}E_\theta(\mathbf{z}_{1:T}, \mathbf{a}^{(k)}).
$$

当前默认使用有限差分估计。对动作维度 $d$ 的单位基向量 $\mathbf{e}_d$，有：

$$
\frac{\partial E}{\partial a_d}
\approx
\frac{
E_\theta(\mathbf{z}_{1:T}, \mathbf{a}+h\mathbf{e}_d)
-
E_\theta(\mathbf{z}_{1:T}, \mathbf{a}-h\mathbf{e}_d)
}{2h}.
$$

代码中对每个动作维度计算上述差分，再扩展到完整动作 chunk。

### 11.4 梯度裁剪

为防止局部能量梯度过大，代码对梯度范数进行裁剪。若：

$$
\|\mathbf{g}_E\|_2 > c,
$$

则：

$$
\mathbf{g}_E \leftarrow \mathbf{g}_E \cdot \frac{c}{\|\mathbf{g}_E\|_2}.
$$

其中 $c$ 是梯度裁剪阈值。

### 11.5 Langevin 更新

初始动作为：

$$
\mathbf{a}^{(0)} = \mathbf{a}_0.
$$

第 $k$ 步更新为：

$$
\mathbf{a}^{(k+1)}
=
\mathbf{a}^{(k)}
-
\frac{\epsilon_k}{2}\lambda_t
\left(
\mathbf{g}_E^{(k)} + \kappa(\mathbf{a}^{(k)}-\mathbf{a}_0)
\right)
+
\sigma\sqrt{\epsilon_k}\boldsymbol{\xi}^{(k)},
$$

其中：

$$
\boldsymbol{\xi}^{(k)}\sim\mathcal{N}(0,I),
$$

$\sigma$ 对应 `noise_scale`。当前 `safe_balanced.yaml` 中：

$$
\sigma=0,
$$

因此实际推理是确定性的局部能量下降，而不是随机采样。

每一步后动作会被裁剪到全局动作范围：

$$
\mathbf{a}^{(k+1)} \leftarrow \mathrm{clip}(\mathbf{a}^{(k+1)}, -a_{\mathrm{clip}}, a_{\mathrm{clip}}).
$$

默认动作范围通常是：

$$
a_{\mathrm{clip}}=1.0.
$$

### 11.6 门控项的作用

门控 $\lambda_t$ 直接缩放能量下降和先验保持项：

$$
\Delta \mathbf{a}^{(k)}
\propto
-\lambda_t
\left(
\nabla_{\mathbf{a}}E_\theta + \kappa(\mathbf{a}^{(k)}-\mathbf{a}_0)
\right).
$$

因此：

- 当 $\lambda_t\approx 0$ 时，CE-AIS 几乎不改变 VLA 动作；
- 当 $\lambda_t$ 较大时，CE-WM 能量景观对动作有更强影响；
- 由于仍有 $\kappa(\mathbf{a}-\mathbf{a}_0)$ 项，动作不会无限偏离 VLA 先验。

---

## 12. 信任域约束

Langevin 得到候选动作 $\tilde{\mathbf{a}}$ 后，Safe CE-AIS 使用逐元素信任域投影：

$$
\mathbf{a}^{\star}
= \mathbf{a}_0 + \mathrm{clip}(\tilde{\mathbf{a}}-\mathbf{a}_0, -\delta_{\max}, \delta_{\max}).
$$

等价地，对每个 batch、时间步和动作维度：

$$
a^{\star}_{b,\tau,d}
= a_{0,b,\tau,d}
+ \min\left(
\max(\tilde{a}_{b,\tau,d}-a_{0,b,\tau,d}, -\delta_{\max}),
\delta_{\max}
\right).
$$

当前 `safe_balanced.yaml` 中：

$$
\delta_{\max}=0.08.
$$

信任域保证：

$$
\|\mathbf{a}^{\star}-\mathbf{a}_0\|_{\infty} \leq \delta_{\max}.
$$

这一步是 Safe CE-AIS 的关键，因为 FLOWER 等强 VLA 已经具有较高动作质量，CE-AIS 不应进行大幅远离策略流形的修正。

---

## 13. 接受/拒绝机制

### 13.1 能量前后评估

在干预前，CE-WM 对 VLA 原始动作计算能量：

$$
E_{\mathrm{before}} = E_\theta(\mathbf{z}_{1:T}, \mathbf{a}_0).
$$

在得到信任域内候选动作后，若候选动作数值有限，则计算：

$$
E_{\mathrm{after}} = E_\theta(\mathbf{z}_{1:T}, \mathbf{a}^{\star}).
$$

否则令：

$$
E_{\mathrm{after}} = E_{\mathrm{before}}.
$$

### 13.2 接受条件

Safe CE-AIS 接受候选动作当且仅当以下条件同时成立：

1. 数值有限：

$$
\mathrm{Finite}(\mathbf{a}^{\star}) = \mathrm{True}.
$$

2. 信任域安全：

$$
\|\mathbf{a}^{\star}-\mathbf{a}_0\|_{\infty}
\leq
\delta_{\max}.
$$

3. 能量改进：

$$
E_{\mathrm{after}}
\leq
E_{\mathrm{before}} - m_{\mathrm{acc}}.
$$

其中 $m_{\mathrm{acc}}$ 对应 `accept_energy_margin`。当前 `safe_balanced.yaml` 中：

$$
m_{\mathrm{acc}} = 10^{-4}.
$$

综合写为：

$$
\mathrm{Accept}
=
\mathbf{1}
\left[
\mathrm{Finite}(\mathbf{a}^{\star})
\land
\|\mathbf{a}^{\star}-\mathbf{a}_0\|_{\infty}\leq\delta_{\max}
\land
E_{\mathrm{after}}\leq E_{\mathrm{before}}-m_{\mathrm{acc}}
\right].
$$

如果关闭 accept/reject，则能量改进条件可被跳过，但当前 Safe CE-AIS 默认启用。

### 13.3 输出规则

若接受：

$$
\mathbf{a}_{\mathrm{exec}} = \mathbf{a}^{\star},
\quad
\mathrm{status}=\mathrm{accepted},
\quad
\mathrm{intervened}=\mathrm{True}.
$$

若拒绝：

$$
\mathbf{a}_{\mathrm{exec}} = \mathbf{a}_0,
\quad
\mathrm{status}=\mathrm{rejected},
\quad
\mathrm{intervened}=\mathrm{False}.
$$

拒绝原因可写为：

$$
\mathrm{reason} =
\begin{cases}
\mathrm{non\_finite\_action}, & \text{if } \neg\mathrm{Finite}(\mathbf{a}^{\star}), \\
\mathrm{trust\_region\_violation}, & \text{if } \|\mathbf{a}^{\star}-\mathbf{a}_0\|_{\infty}>\delta_{\max}, \\
\mathrm{energy\_not\_improved}, & \text{otherwise}.
\end{cases}
$$

由于实现中已经在 steering 后执行信任域投影，实际出现 trust-region violation 的概率很低，但仍作为安全检查保留。

---

## 14. 异常回退机制

CE-AIS 具有两层异常回退。

### 14.1 VLA 预测失败

如果 VLA 预测失败，且存在上一帧动作 $\mathbf{a}_{t-1}$，则返回上一帧动作：

$$
\mathbf{a}_{\mathrm{exec}} = \mathbf{a}_{t-1}.
$$

若没有上一帧动作，则抛出异常。

### 14.2 Steering 失败

如果 encoder、CE-WM、不确定性估计或 steering 过程失败，则回退到当前 VLA 原始动作：

$$
\mathbf{a}_{\mathrm{exec}} = \mathbf{a}_0,
\quad
\mathrm{status}=\mathrm{fallback}.
$$

该设计保证 CE-AIS 作为附加控制层时不会轻易中断基础 VLA 策略。

---

## 15. 可选 rerank 模式

当前主要模式是 `mode="langevin"`。代码中还保留了 rerank-compatible hook。若启用 rerank，可以将动作选择写为候选集最小化：

$$
\mathcal{C} = \{\mathbf{a}_0, \mathbf{a}_0+\boldsymbol{\epsilon}_1, \ldots, \mathbf{a}_0+\boldsymbol{\epsilon}_K\},
$$

其中每个扰动候选都经过信任域投影：

$$
\mathbf{c}_k \leftarrow \Pi_\Delta(\mathbf{c}_k;\mathbf{a}_0).
$$

候选评分为：

$$
S(\mathbf{c}_k)
= E_\theta(\mathbf{z}_{1:T}, \mathbf{c}_k)
+ \rho\|\mathbf{c}_k-\mathbf{a}_0\|_2^2,
$$

其中 $\rho$ 对应 `deviation_weight`。最终选择：

$$
\mathbf{a}^{\star}
= \arg\min_{\mathbf{c}_k\in\mathcal{C}} S(\mathbf{c}_k).
$$

该模式更像能量验证器/动作选择器，而不是梯度控制器。当前主实验仍以 Langevin 模式为主。

---

## 16. 完整推理算法

### Algorithm 1: Safe CE-AIS Test-Time Steering

输入：当前观测 $\mathbf{o}_t$，语言指令 $g$，冻结 VLA $\pi_{\mathrm{VLA}}$，冻结 encoder $f_\phi$，冻结 CE-WM $E_\theta$，门控器 $\Gamma$，steering 模块 $\mathcal{S}$。

输出：最终动作 $\mathbf{a}_{\mathrm{exec}}$ 与诊断信息 $\mathcal{I}_t$。

1. 使用 VLA 生成动作先验：

$$
\mathbf{a}_0 \leftarrow \pi_{\mathrm{VLA}}(\mathbf{o}_t,g).
$$

2. 编码当前观测：

$$
\mathbf{z}_t \leftarrow f_\phi(\mathbf{o}_t).
$$

若 $\|\mathbf{z}_t\|_2 < 10^{-8}$，视为编码异常并回退。

3. 展开 latent sequence：

$$
\mathbf{z}_{1:T} \leftarrow [\mathbf{z}_t,\ldots,\mathbf{z}_t].
$$

4. 计算原始能量：

$$
E_{\mathrm{before}} \leftarrow E_\theta(\mathbf{z}_{1:T},\mathbf{a}_0).
$$

5. 估计不确定性：

$$
u_t \leftarrow \mathrm{Var}_{m=1}^{M}[E_\theta^{(m)}(\mathbf{z}_{1:T},\mathbf{a}_0)].
$$

6. 若存在硬阈值且 $u_t>u_{\max}$，则：

$$
\mathbf{a}_{\mathrm{exec}}\leftarrow \mathbf{a}_0,
\quad
\mathrm{status}\leftarrow\mathrm{abstained\_uncertainty}.
$$

7. 计算门控强度：

$$
\lambda_t \leftarrow \Gamma(u_t;\mathcal{H}_u).
$$

8. 若 $\max(\lambda_t)\leq\lambda_{\min}$，则：

$$
\mathbf{a}_{\mathrm{exec}}\leftarrow \mathbf{a}_0,
\quad
\mathrm{status}\leftarrow\mathrm{vla\_prior}.
$$

9. 执行动作空间 Langevin steering：

$$
\tilde{\mathbf{a}} \leftarrow \mathcal{S}(\mathbf{a}_0,\mathbf{z}_t,E_\theta,\lambda_t).
$$

10. 信任域投影：

$$
\mathbf{a}^{\star}\leftarrow \mathbf{a}_0+\mathrm{clip}(\tilde{\mathbf{a}}-\mathbf{a}_0,-\delta_{\max},\delta_{\max}).
$$

11. 计算动作偏移：

$$
\Delta_a = \|\mathbf{a}^{\star}-\mathbf{a}_0\|_{\infty}.
$$

12. 若 $\mathbf{a}^{\star}$ 有限，则计算：

$$
E_{\mathrm{after}}\leftarrow E_\theta(\mathbf{z}_{1:T},\mathbf{a}^{\star}).
$$

13. 判断接受：

$$
\mathrm{Accept}=\mathbf{1}\left[
\mathrm{Finite}(\mathbf{a}^{\star})
\land
\Delta_a\leq\delta_{\max}
\land
E_{\mathrm{after}}\leq E_{\mathrm{before}}-m_{\mathrm{acc}}
\right].
$$

14. 若接受：

$$
\mathbf{a}_{\mathrm{exec}}\leftarrow\mathbf{a}^{\star}.
$$

否则：

$$
\mathbf{a}_{\mathrm{exec}}\leftarrow\mathbf{a}_0.
$$

15. 返回动作和诊断信息：

$$
(\mathbf{a}_{\mathrm{exec}},\mathcal{I}_t).
$$

---

## 17. 诊断指标

CE-AIS 在 topology 层累积轻量级诊断计数。令总步数为：

$$
N = \mathrm{steps\_total}.
$$

接受率：

$$
r_{\mathrm{accept}} = \frac{N_{\mathrm{accepted}}}{N}.
$$

拒绝率：

$$
r_{\mathrm{reject}} = \frac{N_{\mathrm{rejected}}}{N}.
$$

不确定性放弃率：

$$
r_{\mathrm{abstain}} = \frac{N_{\mathrm{abstained\_uncertainty}}}{N}.
$$

VLA 先验保留率：

$$
r_{\mathrm{prior}} = \frac{N_{\mathrm{vla\_prior}}}{N}.
$$

异常回退率：

$$
r_{\mathrm{fallback}} = \frac{N_{\mathrm{fallback}}}{N}.
$$

平均动作无穷范数偏移：

$$
\bar{\Delta}_a = \frac{1}{N}\sum_{t=1}^{N}\|\mathbf{a}_t^{\star}-\mathbf{a}_{0,t}\|_{\infty}.
$$

平均干预前能量：

$$
\bar{E}_{\mathrm{before}} = \frac{1}{N}\sum_{t=1}^{N}E_{\mathrm{before},t}.
$$

平均干预后能量：

$$
\bar{E}_{\mathrm{after}} = \frac{1}{N}\sum_{t=1}^{N}E_{\mathrm{after},t}.
$$

平均不确定性：

$$
\bar{u} = \frac{1}{N}\sum_{t=1}^{N}u_t.
$$

平均门控强度：

$$
\bar{\lambda} = \frac{1}{N}\sum_{t=1}^{N}\lambda_t.
$$

这些指标用于回答 CE-AIS 是否真的在执行选择性干预，而不是盲目覆盖基础策略。

---

## 18. 长序列任务评估指标

在 CALVIN ABC→D 评估中，每条 chain 包含最多 $L=5$ 个连续子任务。令第 $i$ 条 chain 完成的连续任务数为：

$$
c_i \in \{0,1,2,3,4,5\}.
$$

### 18.1 Lk 成功率

Lk 表示完成至少前 $k$ 个连续任务的比例：

$$
\mathrm{L}k = \frac{1}{N_{\mathrm{chain}}}\sum_{i=1}^{N_{\mathrm{chain}}}\mathbf{1}[c_i\geq k].
$$

例如：

$$
\mathrm{L1}=P(c_i\geq1),
\quad
\mathrm{L5}=P(c_i=5).
$$

### 18.2 平均完成长度

平均完成任务数为：

$$
\bar{c} = \frac{1}{N_{\mathrm{chain}}}\sum_{i=1}^{N_{\mathrm{chain}}}c_i.
$$

### 18.3 条件成功率

条件成功率刻画在前序任务已成功的条件下，第 $k$ 个任务继续成功的概率：

$$
P_k
= P(c_i\geq k \mid c_i\geq k-1)
= \frac{\sum_i \mathbf{1}[c_i\geq k]}{\sum_i \mathbf{1}[c_i\geq k-1]}.
$$

该指标比单独 Lk 更能分析长程错误传播。

---

## 19. 当前推荐配置的数学含义

当前主要实验使用 `configs/safe_balanced.yaml`。其方法含义如下。

### 19.1 Steering 参数

$$
K_{\mathrm{steps}}=2
$$

表示每一步只做两次局部动作空间更新，避免大幅修改 VLA 策略。

$$
\epsilon_0=0.006, \quad \alpha=0.5
$$

表示两步步长分别为：

$$
\epsilon_0=0.006,
\quad
\epsilon_1=0.003.
$$

$$
\sigma=0
$$

表示不注入随机噪声，控制更稳定。

$$
\kappa=25.0
$$

表示强约束动作保持在 VLA 先验附近。

$$
\delta_{\max}=0.08
$$

表示每个动作维度最多偏离 VLA 原始输出 0.08。

$$
m_{\mathrm{acc}}=10^{-4}
$$

表示只有当 CE-WM 能量至少下降 $10^{-4}$ 时才接受校正。

### 19.2 Gating 参数

$$
\lambda_{\max}=0.4,
\quad
\sigma_u=0.1,
\quad
W=50,
\quad
M=5.
$$

表示最大干预幅度中等，门控对不确定性偏移较敏感，并用 5 次 MC-Dropout 估计不确定性。

---

## 20. CE-AIS 与普通 VLA 推理的区别

普通 VLA 推理直接执行：

$$
\mathbf{a}_{\mathrm{exec}} = \pi_{\mathrm{VLA}}(\mathbf{o}_t,g).
$$

CE-AIS 推理执行：

$$
\mathbf{a}_{\mathrm{exec}}
= \begin{cases}
\mathbf{a}^{\star}, & \text{if accepted}, \\
\mathbf{a}_0, & \text{otherwise}.
\end{cases}
$$

其中 $\mathbf{a}^{\star}$ 由冻结 CE-WM 的能量梯度、门控不确定性和信任域约束共同决定。

因此 CE-AIS 并不是重新训练一个策略，也不是替换 VLA，而是在测试时给 VLA 增加一个可拒绝的动作验证与局部恢复层。

---

## 21. 方法贡献的论文式表述

CE-AIS 的方法贡献可归纳为四点。

### 21.1 冻结模型上的测试时主动推理

CE-AIS 不改变 VLA、encoder 或 CE-WM 参数：

$$
\nabla \psi_{\mathrm{VLA}}=0,
\quad
\nabla \phi=0,
\quad
\nabla \theta=0.
$$

测试时唯一被优化的是动作变量：

$$
\nabla_{\mathbf{a}}E_\theta(\mathbf{z},\mathbf{a}) \neq 0.
$$

这使得方法可以作为任意 VLA 的外部控制层。

### 21.2 目标条件能量验证器

CE-WM 学习状态-动作合法性评分：

$$
E_\theta(\mathbf{z},\mathbf{a})
\approx
-\log p_{\mathrm{expert}}(\mathbf{a}\mid\mathbf{z}) + C.
$$

虽然训练不是直接最大似然，但 NCE 使得专家动作能量低于扰动动作：

$$
E_\theta(\mathbf{z},\mathbf{a}_+)
<
E_\theta(\mathbf{z},\mathbf{a}_-).
$$

因此 CE-WM 可作为动作 verifier。

### 21.3 信任域动作选择

CE-AIS 不允许任意能量下降，而是限制在 VLA 先验邻域中：

$$
\mathbf{a}\in\mathcal{B}_\infty(\mathbf{a}_0,\delta_{\max})
=\{\mathbf{a}:\|\mathbf{a}-\mathbf{a}_0\|_\infty\leq\delta_{\max}\}.
$$

因此优化问题可写为：

$$
\min_{\mathbf{a}\in\mathcal{B}_\infty(\mathbf{a}_0,\delta_{\max})}
E_\theta(\mathbf{z},\mathbf{a})
+\frac{\kappa}{2}\|\mathbf{a}-\mathbf{a}_0\|_2^2.
$$

### 21.4 可拒绝的安全干预

CE-AIS 的最终控制律是可拒绝的：

$$
\mathbf{a}_{\mathrm{exec}}
= \mathbf{a}_0
+ \mathrm{Accept}\cdot(\mathbf{a}^{\star}-\mathbf{a}_0).
$$

其中 $\mathrm{Accept}\in\{0,1\}$。这意味着当 CE-WM 判断校正不可靠时，系统保留原始 VLA 行为。

---

## 22. 与实验结果解释的关系

基于上述机制，CE-AIS 的合理实验主张应是：

1. 在 clean 环境中，CE-AIS 应尽量保持 strong VLA baseline，不应大幅破坏原始策略；
2. 在轻中度 OOD 或部分视觉扰动中，CE-WM 能量可能提供有用的局部恢复方向；
3. 在严重 physics OOD 中，若 VLA 和 CE-WM 都处于训练分布外，CE-AIS 可能选择回退或仍然失败；
4. 因此 CE-AIS 的核心指标不仅是成功率，还包括接受率、拒绝率、能量下降、动作偏移和不确定性。

这与 Safe CE-AIS 的设计一致：CE-AIS 不是保证所有 OOD 都提升的万能控制器，而是冻结 VLA 上的选择性、可拒绝、受信任域约束的动作空间恢复模块。

---

## 23. 可直接用于论文的方法段落

给定观测 $\mathbf{o}_t$ 和语言目标 $g$，我们首先通过冻结的视觉语言动作模型生成动作先验 $\mathbf{a}_0=\pi_{\mathrm{VLA}}(\mathbf{o}_t,g)$。随后，对比编码器将当前观测映射为单位归一化潜在状态 $\mathbf{z}_t=f_\phi(\mathbf{o}_t)$，并将其复制为长度为 $T$ 的状态序列 $\mathbf{z}_{1:T}$。冻结的因果能量世界模型 $E_\theta$ 对状态-动作序列 $(\mathbf{z}_{1:T},\mathbf{a}_0)$ 计算能量 $E_{\mathrm{before}}$，并通过 MC-Dropout 估计 epistemic uncertainty $u_t$。不确定性经由历史自适应 Gaussian gate 转换为干预强度 $\lambda_t$。若不确定性过高或门控强度过低，系统直接执行 VLA 原始动作。

当 CE-AIS 决定干预时，我们在动作空间内执行有限步退火 Langevin 更新：

$$
\mathbf{a}^{(k+1)}
=
\mathbf{a}^{(k)}
-
\frac{\epsilon_0\alpha^k}{2}\lambda_t
\left(
\nabla_{\mathbf{a}}E_\theta(\mathbf{z}_{1:T},\mathbf{a}^{(k)})
+\kappa(\mathbf{a}^{(k)}-\mathbf{a}_0)
\right)
+
\sigma\sqrt{\epsilon_0\alpha^k}\boldsymbol{\xi}^{(k)}.
$$

随后将候选动作投影到 VLA 先验的 $\ell_\infty$ 信任域：

$$
\mathbf{a}^{\star}
=
\mathbf{a}_0+	ext{clip}(\mathbf{a}^{(K)}-\mathbf{a}_0,-\delta_{\max},\delta_{\max}).
$$

最终仅当候选动作数值有限、满足信任域约束且 CE-WM 能量下降时接受校正：

$$
\mathrm{Accept}=\mathbf{1}
\left[
\mathrm{Finite}(\mathbf{a}^{\star})
\land
\|\mathbf{a}^{\star}-\mathbf{a}_0\|_{\infty}\leq\delta_{\max}
\land
E_\theta(\mathbf{z}_{1:T},\mathbf{a}^{\star})
\leq
E_\theta(\mathbf{z}_{1:T},\mathbf{a}_0)-m_{\mathrm{acc}}
\right].
$$

最终执行动作写为：

$$
\mathbf{a}_{\mathrm{exec}}
=
\mathbf{a}_0+\\mathrm{Accept}\cdot(\mathbf{a}^{\star}-\mathbf{a}_0).
$$

该形式强调 CE-AIS 是一个测试时动作空间校正器：它不改变基础 VLA 的参数，也不改变 CE-WM 参数，而是在冻结模型的能量景观下对 VLA 输出进行小范围、可拒绝的局部优化。

---

## 24. 代码模块对应关系

| 方法组件 | 代码位置 | 数学对象 |
|---|---|---|
| VLA adapter | `src/dual_stream/vla_adapter.py` | $\pi_{\mathrm{VLA}}$ |
| 双流拓扑 | `src/dual_stream/topology.py` | 完整推理流程 $\mathcal{A}\circ\Pi_\Delta\circ\mathcal{S}$ |
| 对比编码器 | `src/encoders/contrastive_encoder.py` | $f_\phi$ |
| CE-WM | `src/world_model/ce_wm.py` | $E_\theta$ |
| InfoNCE / NCE loss | `src/training/losses.py` | $\mathcal{L}_{\mathrm{InfoNCE}},\mathcal{L}_{\mathrm{CEWM}}$ |
| 预训练管线 | `src/training/pretrain_pipeline.py` | encoder/CE-WM 训练过程 |
| 不确定性门控 | `src/steering/bilateral_gating.py` | $\Gamma(u_t;\mathcal{H}_u)$ |
| EFE steering | `src/steering/efe_steering.py` | $\mathcal{S}$, $\Pi_\Delta$ |
| Langevin 更新 | `src/steering/langevin.py` | 动作空间退火 Langevin 公式 |
| 负样本构造 | `src/data/perturbation.py` | $P_k(\mathbf{a}_+)$ |

---

## 25. 总结

CE-AIS 可以被形式化为一个由四个冻结组件和一个测试时动作优化器组成的系统：

$$
\boxed{
\mathbf{a}_{\mathrm{exec}}
=
\mathbf{a}_0
+
\mathbf{1}[\mathrm{safe\ energy\ improvement}]
\cdot
\left(
\Pi_\Delta\left(
\mathcal{S}(\mathbf{a}_0,\mathbf{z}_t,E_\theta,\lambda_t)
\right)
-
\mathbf{a}_0
\right)
}
$$

其中：

$$
\mathbf{a}_0=\pi_{\mathrm{VLA}}(\mathbf{o}_t,g),
\quad
\mathbf{z}_t=f_\phi(\mathbf{o}_t),
\quad
\lambda_t=\Gamma(\mathrm{Var}_{\mathrm{dropout}}[E_\theta(\mathbf{z},\mathbf{a}_0)]).
$$

这一公式概括了当前代码实现的本质：CE-AIS 不是训练新的策略，而是利用预训练能量模型在测试时验证并微调 VLA 动作；不是无条件覆盖基础模型，而是通过不确定性门控、信任域约束和能量接受/拒绝机制实现选择性干预。该框架特别适合被描述为 model-agnostic、frozen-backbone、test-time action-space active inference controller。
