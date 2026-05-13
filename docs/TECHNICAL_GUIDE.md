# CE-AIS 技术文档

## 一、项目概述

CE-AIS（Causal-Energy Active Inference Steering，基于因果能量主动推理的梯度免更新动作引导框架）是一个面向具身智能（Embodied AI）的测试时自适应（Test-Time Adaptation）系统。

核心创新：在推理阶段 **100% 冻结所有网络参数**，不执行任何反向传播参数更新。通过基于主动推理（Active Inference）理论的期望自由能（EFE）最小化，利用退火朗之万动力学（Annealed Langevin Dynamics）在动作空间中实时偏转候选动作，使其符合物理因果律约束。

### 解决的问题

当前 VLA（Vision-Language-Action）模型在面对分布外（OOD）场景时极为脆弱——光照变化、物体位姿偏移、摩擦力突变等都会导致灾难性失败。现有方法（如 AdaWorldPolicy、TT-VLA）通过在线微调参数来适应，但这带来：
- 计算延迟（反向传播阻塞高频控制循环）
- 灾难性遗忘（频繁参数覆写破坏预训练知识）

CE-AIS 的解法：不改参数，改动作。用能量世界模型评估动作的物理合法性，用朗之万动力学把不合法的动作"推"回安全区域。

### 与现有方法的本质区别

| 维度 | AdaWorldPolicy / TT-VLA | CE-AIS |
|------|------------------------|--------|
| 测试时更新什么 | 网络参数（LoRA/梯度） | 动作输出张量（非参数） |
| 世界模型用途 | 生成预测 → 计算误差 → 更新权重 | 评估能量 → 引导动作偏转 |
| 灾难性遗忘风险 | 高 | 零（参数完全冻结） |
| 推理延迟 | 高（需反向传播） | 极低（3-5步朗之万迭代） |
| 理论基础 | 预测编码 | 主动推理 + 能量模型 |

---

## 二、系统架构

### 非对称双流拓扑

```
┌─────────────────────────────────────────────────────────┐
│                    输入层                                 │
│  多模态观测 o_t (RGB + Depth + Pose) + 语言指令 l        │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌─────────────────────┐
│ 主控语义策略流    │   │ 对比编码器           │
│ VLA 基座模型      │   │ ContrastiveEncoder  │
│ (冻结, 数十亿参数) │   │ (RGB+Depth+Pose→z_t)│
│                  │   └──────────┬──────────┘
│ 输出: 候选动作 a_0│              │ 潜变量 z_t
└────────┬─────────┘              │
         │                        ▼
         │            ┌─────────────────────┐
         │            │ 因果能量裁判流       │
         │            │ Mamba-3 CE-WM       │
         │            │ (冻结, 100-300M参数) │
         ├───────────►│                     │
         │            │ 输出: 能量 E + 不确定性 u_t
         │            └──────┬──────┬───────┘
         │                   │      │
         │                   ▼      ▼
         │            ┌──────────────────┐
         │            │ 双向不确定性门控  │
         │            │ λ(u_t) = λ_max · │
         │            │ exp(-(u_t-μ)²/2σ²)│
         │            └────────┬─────────┘
         │                     │ 引导强度 λ
         ▼                     ▼
┌──────────────────────────────────┐
│ EFE 引导偏转                      │
│ 退火朗之万动力学 (3-5步)           │
│ a_{k+1} = a_k - ε/2·λ·(∇E + KL) │
│ + √ε · η                         │
│                                   │
│ 仅修改动作张量，不修改网络参数      │
└──────────────┬───────────────────┘
               │
               ▼
        校正后动作 a* → 机械臂执行器
```

### 数据流时序

1. 环境产生观测 `o_t = (RGB, Depth, Pose)`
2. VLA 基座接收观测+语言指令，输出候选动作 `a_0`
3. 对比编码器将观测压缩为潜变量 `z_t`
4. CE-WM 接收 `(z_t, a_0)`，输出能量值 `E` 和认知不确定性 `u_t`
5. 双向门控根据 `u_t` 计算引导强度 `λ`
6. 朗之万动力学执行 3-5 步迭代，偏转 `a_0` → `a*`
7. `a*` 发送给机械臂执行

---

## 三、核心模块详解

### 3.1 对比预训练编码器 (`src/encoders/`)

**作用**：将高维多模态观测压缩为低维潜变量，供 CE-WM 在紧凑空间中追踪物理状态。

**架构**：
- 视觉骨干：ResNet-18（默认）或 ViT-Small，RGB 和深度图共享骨干
- 深度图适配：1→3 通道 Conv2d，兼容 RGB 骨干
- 本体姿态 MLP：7维姿态 → 128维特征
- 融合头：拼接 RGB+Depth+Pose 特征 → LayerNorm → L2 归一化

**关键配置** (`configs/base.yaml` → `encoder` 节)：
```yaml
encoder:
  backbone_type: "resnet18"    # 骨干网络类型
  latent_dim: 128              # 潜变量维度 d_z
  pose_dim: 7                  # 末端执行器姿态维度
  temperature: 0.07            # InfoNCE 温度参数
  image_size: [200, 200]       # 输入图像分辨率
```

**性能调优**：
- `latent_dim` 越大表达能力越强，但 CE-WM 推理开销也越大。128 是平衡点
- `temperature` 越小对比学习越"尖锐"，0.07 是经验最优值
- 使用 ViT-Small 骨干可能在复杂场景下表现更好，但推理更慢

### 3.2 Mamba-3 因果能量世界模型 (`src/world_model/`)

**作用**：在抽象潜空间中追踪物理状态演化，输出标量能量值评估动作的物理合法性。

**架构**：
- 输入投影：`(d_z + d_a) → d_model` 线性层
- 时序引擎：N 层 Mamba-3 Block 堆叠
  - 复数域状态更新（等效 RoPE）
  - MIMO 矩阵并行输出
  - 指数梯形离散化（无因果卷积）
- 能量头：`d_model → d_model/2 → 1` MLP

**Mamba-3 核心特性**：
- O(1) 内存复杂度（vs Transformer 的 O(n²)）
- 复数域状态追踪：长时序不遗忘
- MIMO 并行：每次内存读取执行 4x+ FLOPs

**关键配置** (`configs/base.yaml` → `ce_wm` 节)：
```yaml
ce_wm:
  d_model: 512                # 隐藏维度
  d_state: 64                 # SSM 状态维度
  n_layers: 24                # Mamba-3 层数
  expand_factor: 2            # 扩展因子
  mimo_groups: 4              # MIMO 通道组数
  action_dim: 7               # 动作维度
  dropout: 0.1                # MC-Dropout 概率
```

**参数量控制**：目标 100M-300M。调节 `d_model`、`n_layers`、`expand_factor` 的组合：
- 轻量级（~100M）：d_model=512, n_layers=32, expand=2
- 标准（~200M）：d_model=640, n_layers=36, expand=3
- 重量级（~300M）：d_model=768, n_layers=40, expand=3

**能量景观训练**：
- 正样本：CALVIN 专家演示轨迹（低能量）
- 负样本：对专家动作施加对抗性摄动（高能量）
  - 速度矢量反转
  - 夹爪状态异常
  - 随机大幅度位移
- 损失函数：NCE/InfoNCE

### 3.3 EFE 引导偏转 (`src/steering/`)

**作用**：在推理阶段通过退火朗之万动力学偏转候选动作。

**核心公式**：
```
a_{k+1} = a_k - (ε_k/2) · λ · [∇_a E(z_t, a_k) + kl_weight · (a_k - a_0)] + √ε_k · η
```

- `ε_k = ε_0 · α^k`：退火步长（随迭代衰减）
- `λ`：双向门控引导强度
- `∇_a E`：能量梯度（仅对动作张量求导，不对网络参数求导）
- `kl_weight · (a_k - a_0)`：KL 散度约束（不过度偏离 VLA 先验）
- `η ~ N(0, I)`：探索噪声

**关键配置** (`configs/base.yaml` → `steering` 节)：
```yaml
steering:
  n_steps: 5                  # 朗之万迭代步数
  step_size: 0.01             # 初始步长 ε_0
  anneal_rate: 0.5            # 退火率 α
  noise_scale: 0.001          # 探索噪声强度
  kl_weight: 10.0             # KL 散度权重
```

**性能调优**：
- `n_steps`：3-5 步通常足够。增加步数提升偏转精度但增加延迟
- `step_size`：太大导致动作跳跃，太小偏转不够。0.01 是安全起点
- `kl_weight`：越大越保守（更接近 VLA 原始输出），越小越激进（更依赖能量引导）
- `noise_scale`：增加探索性，但太大会引入抖动

### 3.4 双向不确定性门控 (`src/steering/bilateral_gating.py`)

**作用**：根据 CE-WM 的认知不确定性动态调节能量引导强度，防止极端 OOD 下错误梯度毒化动作流。

**核心公式**：
```
λ(u_t) = λ_max · exp(-(u_t - μ_u)² / (2·σ_u²))
```

**行为**：
- 低不确定性（CE-WM 自信）→ λ → λ_max → 强力能量纠偏
- 高不确定性（CE-WM 也蒙了）→ λ → 0 → 回退至 VLA 保守先验

**关键配置**：
```yaml
bilateral_gating:
  lambda_max: 1.0             # 最大引导强度
  sensitivity: 0.1            # 门控灵敏度 σ_u
  window_size: 50             # 历史窗口长度
  mc_samples: 5               # MC-Dropout 采样次数
```

### 3.5 CALVIN 仿真集成 (`src/evaluation/`)

**支持的评估协议**：
- ABC→D 零样本跨环境多任务链式评估
- 单任务成功率统计

**OOD 干扰注入**：
- 物理干扰：质量密度变异、摩擦系数修改
- 视觉干扰：光照闪烁、高斯噪点、相机姿态偏转

**评估指标**：
- 多任务链式成功率（1-5 连击）
- 瞬态恢复时间（环境突变后胜率恢复速度）
- 轨迹力学平滑度（Jerk = 三阶数值差分）
- 计算延时帕累托边界（延时 vs 成功率）

---

## 四、运行指南

### 4.1 环境准备

```bash
cd ~/yejinxuan/ce-ais
uv sync
```

验证环境：
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出: 2.5.1+cu124 True
```

### 4.2 准备 CALVIN 数据

将 CALVIN 数据集放置在 `data/` 目录下：
```bash
# 下载 CALVIN 数据集（参考 CALVIN 官方仓库）
# 数据结构应为:
# data/
#   calvin/
#     training/
#       A/ B/ C/ D/
#     validation/
```

### 4.3 阶段一：预训练

预训练分两个阶段：先训练对比编码器，再训练 CE-WM。

```bash
# 使用默认配置预训练
uv run python scripts/pretrain.py --config configs/base.yaml

# 自定义参数
uv run python scripts/pretrain.py --config configs/base.yaml \
    --override training.batch_size=32 \
    --override training.learning_rate=5e-5 \
    --override training.encoder_epochs=200 \
    --override training.ce_wm_epochs=300

# 从检查点恢复训练
uv run python scripts/pretrain.py --config configs/base.yaml --resume
```

预训练完成后会输出能量分布统计报告，确认正样本平均能量 < 负样本平均能量。

### 4.4 阶段二：评估

```bash
# 标准评估（无 OOD 干扰）
uv run python scripts/evaluate.py --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt

# 注入物理 OOD 干扰
uv run python scripts/evaluate.py --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt \
    --ood physics --ood-strength 1.5

# 注入视觉 OOD 干扰
uv run python scripts/evaluate.py --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt \
    --ood visual --ood-strength 2.0

# 同时注入物理+视觉干扰
uv run python scripts/evaluate.py --config configs/base.yaml \
    --checkpoint checkpoints/best_model.pt \
    --ood physics visual --ood-strength 1.5
```

### 4.5 阶段三：消融实验

```bash
# 列出可用消融变体
uv run python scripts/ablation.py --config configs/base.yaml --list

# 运行所有消融实验
uv run python scripts/ablation.py --config configs/base.yaml

# 运行指定消融
uv run python scripts/ablation.py --config configs/base.yaml \
    --variants no_gating mse_energy
```

三组消融实验：
1. `no_gating`：剥离双向门控 → 验证门控对极端 OOD 的防御必要性
2. `mse_energy`：MSE 重建替换能量判别 → 验证判别式能量的优越性
3. `mamba1_backbone`：Mamba-1 替换 Mamba-3 → 验证 Mamba-3 的效率优势

### 4.6 运行测试

```bash
# 全部测试（48 个）
uv run pytest tests/ -v

# 仅属性测试
uv run pytest tests/property/ -v

# 仅单元测试
uv run pytest tests/unit/ -v

# 仅集成测试
uv run pytest tests/integration/ -v

# 带覆盖率
uv run pytest tests/ -v --cov=src --cov-report=html
```

---

## 五、性能调优指南

### 5.1 显存优化（单卡 RTX 3090/4090 24GB）

| 策略 | 配置 | 效果 |
|------|------|------|
| 减小 CE-WM 规模 | `d_model=384, n_layers=16` | 显存减半，精度略降 |
| 减小批大小 | `training.batch_size=16` | 显存线性降低 |
| 开启混合精度 | `training.amp=true` | 显存减少 ~40% |
| 减小图像分辨率 | `encoder.image_size=[128,128]` | 编码器显存大幅降低 |
| 减少 MC-Dropout 采样 | `bilateral_gating.mc_samples=3` | 推理显存降低 |

### 5.2 推理速度优化

| 策略 | 配置 | 效果 |
|------|------|------|
| 减少朗之万步数 | `steering.n_steps=3` | 延迟降低 40%，偏转精度略降 |
| 减小 CE-WM 层数 | `ce_wm.n_layers=12` | 能量评估加速 2x |
| 减少 MC 采样 | `bilateral_gating.mc_samples=2` | 不确定性估计加速 |
| 使用更小的潜变量 | `encoder.latent_dim=64` | 全链路加速 |

### 5.3 精度优化

| 策略 | 配置 | 效果 |
|------|------|------|
| 增大 CE-WM 规模 | `d_model=768, n_layers=40` | 能量景观更精细 |
| 增加负样本比例 | `training.neg_sample_ratio=10` | 能量边界更清晰 |
| 增加朗之万步数 | `steering.n_steps=8` | 偏转更精确 |
| 降低 KL 权重 | `steering.kl_weight=5.0` | 更激进的能量引导 |
| 增大门控窗口 | `bilateral_gating.window_size=100` | 更稳定的不确定性估计 |

### 5.4 推荐配置方案

**快速验证（开发调试）**：
```yaml
ce_wm: {d_model: 128, n_layers: 4, expand_factor: 2}
encoder: {latent_dim: 32, image_size: [64, 64]}
steering: {n_steps: 2}
training: {batch_size: 8, encoder_epochs: 10, ce_wm_epochs: 20}
```

**标准实验（论文主实验）**：
```yaml
ce_wm: {d_model: 512, n_layers: 24, expand_factor: 2}
encoder: {latent_dim: 128, image_size: [200, 200]}
steering: {n_steps: 5}
training: {batch_size: 64, encoder_epochs: 100, ce_wm_epochs: 200}
```

**极致性能（消融对比）**：
```yaml
ce_wm: {d_model: 768, n_layers: 40, expand_factor: 3}
encoder: {latent_dim: 256, image_size: [200, 200]}
steering: {n_steps: 8}
training: {batch_size: 32, encoder_epochs: 200, ce_wm_epochs: 500, neg_sample_ratio: 10}
```

---

## 六、配置系统

### 6.1 配置文件结构

所有配置集中在 `configs/base.yaml`，分为以下节：

```yaml
project:          # 项目元信息（名称、随机种子、设备）
encoder:          # 对比编码器配置
ce_wm:            # Mamba-3 CE-WM 配置
steering:         # EFE 偏转配置
bilateral_gating: # 双向门控配置
training:         # 预训练配置
evaluation:       # 评估配置（协议、OOD 干扰参数）
logging:          # 日志配置（TensorBoard/W&B）
```

### 6.2 配置继承

消融实验配置通过 `inherit_from` 继承基础配置，仅覆盖变更项：

```yaml
# configs/ablation/no_gating.yaml
inherit_from: "../base.yaml"
bilateral_gating:
  sensitivity: 1e10  # 极大灵敏度 → 门控退化为常数
```

### 6.3 命令行覆盖

任何配置项都可通过命令行覆盖：
```bash
uv run python scripts/pretrain.py --config configs/base.yaml \
    --override ce_wm.d_model=768 \
    --override training.batch_size=32
```

---

## 七、测试体系

### 48 个测试覆盖 13 条正确性属性

| Property | 描述 | 测试文件 |
|----------|------|----------|
| P1 | 组件输入输出形状不变量 | `test_prop_shapes.py` |
| P2 | 对比损失数学性质 | `test_prop_losses.py` |
| P3 | 配置驱动组件实例化 | `test_prop_config.py` |
| P4 | 序列化 round-trip | `test_prop_roundtrip.py` |
| P5 | CE-WM 参数量范围约束 | `test_prop_params.py` |
| P6 | 对抗摄动改变动作 | `test_prop_perturb.py` |
| P7 | 正负样本批次比例 | `test_prop_perturb.py` |
| P8 | 摄动策略注册与检索 | `test_prop_perturb.py` |
| P9 | 推理时参数绝对冻结 | `test_prop_frozen.py` |
| P10 | 能量梯度存在且有限 | `test_prop_gradient.py` |
| P11 | 高斯门控函数行为 | `test_prop_gating.py` |
| P12 | 物理指标计算正确性 | `test_prop_metrics.py` |
| P13 | 配置继承覆盖正确性 | `test_prop_config.py` |

所有属性测试使用 Hypothesis 库，每个属性最少 100 次随机迭代。

---

## 八、文件清单

```
src/
├── encoders/
│   ├── backbone.py              # 视觉骨干工厂（ResNet-18/ViT-Small）
│   └── contrastive_encoder.py   # 多模态对比编码器
├── world_model/
│   ├── mamba3_core.py           # Mamba-3 SSM 核心层
│   ├── ce_wm.py                 # 因果能量世界模型
│   └── energy_head.py           # MLP 能量头
├── steering/
│   ├── langevin.py              # 退火朗之万动力学
│   ├── efe_steering.py          # EFE 引导偏转
│   └── bilateral_gating.py      # 双向不确定性门控
├── dual_stream/
│   ├── topology.py              # 双流推理编排
│   └── vla_adapter.py           # VLA 插件适配器
├── data/
│   ├── calvin_dataset.py        # CALVIN 数据加载
│   ├── perturbation.py          # 对抗性摄动策略
│   └── data_constructor.py      # 正负样本构造
├── training/
│   ├── pretrain_pipeline.py     # 预训练管线
│   └── losses.py                # NCE/InfoNCE 损失
├── evaluation/
│   ├── calvin_integration.py    # CALVIN 环境封装
│   ├── metrics.py               # 评估指标
│   ├── baseline_framework.py    # 基线对比框架
│   └── ablation.py              # 消融实验框架
├── config/
│   ├── config_manager.py        # 配置管理器
│   └── schema.py                # 配置数据类
├── utils/
│   ├── logger.py                # 日志系统
│   ├── checkpoint.py            # 检查点管理
│   └── visualization.py         # 可视化工具
└── data_structures.py           # 核心数据结构定义
```
