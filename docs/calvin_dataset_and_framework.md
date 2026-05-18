# CALVIN 数据集格式与 CE-AIS 框架说明

## 1. CALVIN 数据集是什么

CALVIN（Composing Actions from Language and Vision）是一个面向长时程机器人操作任务的基准数据集。每个 episode 由连续帧组成，每帧存储为一个 `.npz` 文件，记录了机器人在一个时刻的完整感知与控制信息。

本项目使用的数据集为 `task_ABC_D`，训练集约 179 万帧，验证集约 数千帧。

---

## 2. 每帧 .npz 文件的完整字段

每个 `episode_XXXXXXX.npz` 文件包含以下字段：

| 字段名 | Shape | dtype | 实测数值范围 | 含义 |
|--------|-------|-------|-------------|------|
| `rgb_static` | (200, 200, 3) | uint8 | [0, 255] | 固定视角（第三人称俯视）摄像头的 RGB 图像。200×200 分辨率，用于感知整体场景和物体位置 |
| `depth_static` | (200, 200) | float32 | [3.7, 6.2] m | 与 rgb_static 同视角的深度图，单位为米。提供精确的三维几何信息，用于判断空间距离与遮挡 |
| `rgb_gripper` | (84, 84, 3) | uint8 | [0, 254] | 安装在夹爪末端的腕部摄像头 RGB 图像，84×84 分辨率。提供近距离操作细节视角 |
| `depth_gripper` | (84, 84) | float32 | [0.08, 0.69] m | 与 rgb_gripper 同视角的深度图，距离范围更近（夹爪前方约 10cm~70cm） |
| `rgb_tactile` | (160, 120, 6) | uint8 | [93, 244] | 触觉传感器图像，6 通道（左右两个触觉传感器各 3 通道）。在本项目中未使用 |
| `depth_tactile` | (160, 120, 2) | float32 | [0, 0] | 触觉传感器深度信息，当前数据中全为 0，为预留字段 |
| `robot_obs` | (15,) | float64 | [-1.81, 3.12] | 机器人本体感受观测，完整 15 维含义如下（见下方详细说明） |
| `scene_obs` | (24,) | float64 | [-2.48, 0.46] | 场景状态观测，包含场景中各可操作物体（积木、开关等）的位姿信息，共 24 维。**真实部署时无法获取**（需要外部定位系统），本项目不使用 |
| `actions` | (7,) | float64 | [-0.23, 3.12] | 绝对动作指令（TCP 目标位姿 + 夹爪），与 `rel_actions` 的区别在于这是全局坐标系下的目标值 |
| `rel_actions` | (7,) | float64 | [-0.04, 1.00] | **相对动作指令**（增量控制），含义见下方详细说明。VLA 策略输出的即为此格式 |

### robot_obs 15 维详细含义

```
robot_obs[:3]   # TCP 位置 (x, y, z)，单位米，世界坐标系
robot_obs[3:6]  # TCP 姿态 (roll, pitch, yaw)，单位弧度
robot_obs[6]    # 夹爪开合宽度，单位米（0=完全关闭，0.08=完全打开）
robot_obs[7:14] # 7 个关节角度，单位弧度
robot_obs[14]   # 夹爪速度或接触力（附加信息）
```

本项目只使用前 7 维 (`robot_obs[:7]`)，即 TCP 位姿 + 夹爪状态，这是驱动末端执行器所需的最小状态描述。

### rel_actions 7 维详细含义

```
rel_actions[:3] # 位置增量 (Δx, Δy, Δz)，单位米，相对当前 TCP 位置
rel_actions[3:6]# 姿态增量 (Δroll, Δpitch, Δyaw)，单位弧度
rel_actions[6]  # 夹爪动作，{-1=关闭, +1=打开}（二值化控制）
```

相对动作的优势在于与机器人绝对位置无关，泛化性更好，与主流 VLA 策略（如 OpenVLA）的输出格式一致。

---

## 3. CE-AIS 框架是什么

CE-AIS（**C**ausal-**E**nergy **A**ctive **I**nference **S**teering）是一个面向具身智能的**推理时动作校正框架**。

**核心思想**：在不修改底层 VLA 策略参数的前提下，通过一个预训练的因果能量世界模型（CE-WM）评估候选动作的"物理合法性"，并利用能量梯度在推理时迭代修正不合理的动作，提升策略在分布外（OOD）环境下的鲁棒性。

### 工作流程

```
观测 (rgb + depth + pose)
        │
        ▼
  ContrastiveEncoder     ← 多模态感知融合
        │  z ∈ R^128（归一化潜变量）
        ▼
  CausalEnergyWorldModel ← 因果时序建模（Mamba-3，32层）
        │  E ∈ R^1（标量能量值）
        ▼
  Steering（朗之万动力学）← ∇_a E 引导动作偏转
        │
        ▼
  校正后动作 a* → 发送给机器人执行
```

低能量 = 动作符合物理规律、大概率能达成目标；高能量 = 危险或不合理动作。

### 两阶段预训练

**Stage 1：Contrastive Encoder 预训练**
- 目标：学习一个多模态观测到紧凑潜变量 z 的映射
- 方法：InfoNCE 对比学习，同一 episode 内时序相邻帧互为正样本对
- 输入：`rgb_static` + `depth_static` + `robot_obs[:7]`

**Stage 2：Causal Energy World Model 预训练**
- 目标：学习"哪些动作序列是合理的"的能量景观
- 方法：NCE（Noise Contrastive Estimation），真实动作序列为低能量正样本，随机噪声动作为高能量负样本
- 输入：Encoder 编码的 z 序列 + `rel_actions[:7]` 序列（窗口长度 16）

---

## 4. 为什么只使用这四个字段

| 字段 | 用途 | 为何不用其他字段 |
|------|------|-----------------|
| `rgb_static` | 场景语义感知，物体位置 | `rgb_gripper` 视角局限（近距离），不能反映全局场景；`rgb_tactile` 需要额外硬件 |
| `depth_static` | 三维几何，空间距离判断 | `depth_gripper` 范围太近，仅在接触阶段有用，泛化性差 |
| `robot_obs[:7]` | 机器人当前末端状态（TCP 位姿 + 夹爪） | 后 8 维为关节角，对末端任务空间控制冗余；`scene_obs` 真实部署无法获取 |
| `rel_actions[:7]` | 待评估的控制指令 | `actions`（绝对坐标）与机器人绝对位置耦合，泛化性差；相对动作与主流 VLA 输出格式一致 |

**设计原则**：选择真实机器人部署时**都能获取到**的信息。`scene_obs` 需要外部定位系统，`rgb_tactile` / `depth_tactile` 需要特殊触觉传感器，这两类在真实硬件上通常不可用，因此排除。

---

## 5. mmap 格式说明

通过 `scripts/convert_calvin_mmap.py` 将原始 `.npz` 转换为内存映射格式，转换后结构如下：

```
calvin_mmap/
├── training/
│   ├── rgb.npy          # [N, 200, 200, 3] uint8    ← 约 214 GB
│   ├── depth.npy        # [N, 200, 200]   float16   ← 约 107 GB
│   ├── pose.npy         # [N, 7]          float32
│   ├── action.npy       # [N, 7]          float32
│   ├── frame_ids.npy    # [N]             int64     ← 帧编号索引
│   └── ep_start_end_ids.npy               ← episode 边界，用于采样时不跨 episode
└── validation/
    └── ...
```

**为什么转换后更小（353G < 524G）**：
- 原始 `.npz` 保存了所有 10 个字段，转换只提取了 4 个
- 丢弃字段包括：`rgb_gripper`、`depth_gripper`、`rgb_tactile`、`depth_tactile`、`scene_obs`、`actions`（冗余）、`robot_obs` 后 8 维
- `depth` 从 float32 压缩为 float16，精度损失在合理范围内（深度误差 < 1mm）
