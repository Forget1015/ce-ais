# LIBERO 实验启动详细步骤

本文档列出 CE-AIS 接入 LIBERO benchmark 的完整执行步骤，包括每一步需要下载什么、从哪里下载、如何验证。

---

## LIBERO 数据集完整介绍

### 什么是 LIBERO

LIBERO（**Li**felong **Be**nchmark for **Ro**bot Learning）是 NeurIPS 2023 发布的机器人模仿学习 benchmark。它使用 robosuite 仿真器 + Franka Panda 机械臂，在桌面操作场景中提供标准化的任务族、demo 数据和评估协议。

LIBERO 在 VLA 社区（OpenVLA、π0/π0.5、Diffusion Policy）中已成为事实标准评估平台，因为它：
- 任务定义清晰（自然语言指令 → 具体操作目标）
- 有标准 train/eval 协议
- 多个强 baseline 有公开数字
- 支持 spatial/object/goal/long-horizon 多维度分析

### Suite 划分与用途

LIBERO 包含 **5 个子数据集（suite）**，每个设计目的不同：

| Suite | 任务数 | 设计目的 | 控制变量 | 任务举例 |
|-------|--------|---------|---------|---------|
| **libero_spatial** | 10 | 空间关系泛化 | 同一物体（黑碗），不同空间位置 | "pick up the black bowl **between the plate and the ramekin**" |
| **libero_object** | 10 | 物体识别泛化 | 同一动作（放入篮子），不同物体 | "pick up the **alphabet soup** and place it in the basket" |
| **libero_goal** | 10 | 目标多样性 | 不同类型操作目标 | "open the middle drawer", "push the plate" |
| **libero_10** | 10 | 长程多步任务 | 需要多步连续操作 | "turn on the stove **and** put the moka pot on it" |
| **libero_90** | 90 | 大规模预训练/泛化 | 覆盖多场景多物体 | 跨 kitchen/living room/study 场景 |

**关键关系**：
- libero_spatial / object / goal / 10 合称 **"4 standard suites"**（共 40 个 task），用于主评估
- libero_90 的 90 个 task 与 4 standard suites 的 40 个 task **互不重叠**
- libero_90 通常用于预训练，4 standard suites 用于评估

### 各 Suite 详细说明

**libero_spatial（空间关系）**

所有 10 个 task 都是 "pick up the black bowl from [某个位置] and place it on the plate"。
区别只在于黑碗的空间位置（桌子中间、柜子里、盘子旁边、烤碗上面等）。
测试模型是否理解空间关系描述。

**libero_object（物体泛化）**

所有 10 个 task 都是 "pick up [某个物体] and place it in the basket"。
区别只在于物体种类（字母汤罐头、BBQ酱、黄油、奶酪等）。
测试模型是否能识别不同物体并执行相同动作。

**libero_goal（目标多样性）**

10 个 task 有不同类型的操作目标：开抽屉、推盘子、放碗进柜子、关门等。
测试模型是否能理解多样化的语言指令并执行对应动作。

**libero_10（长程任务，也叫 LIBERO-Long）**

10 个 task 都需要两步或多步连续操作（例如"打开灶台 **并且** 把摩卡壶放上去"）。
平均 demo 长度 276 步，是其他 suite 的 2 倍。
**最适合 CE-AIS**：长程任务中动作累积误差大，CE-WM 的纠错机制最有价值。

**libero_90（大规模预训练集）**

90 个 task 分布在 5 类场景：KITCHEN、LIVING_ROOM、STUDY 等。
数据量最大（66.9 万帧），适合训练通用模型或做预训练。
与 4 standard suites 无重叠，可用于训练后在 standard suites 上做 zero-shot 泛化评估。

### 数据格式

每个 task 对应一个 HDF5 文件（`{task_name}_demo.hdf5`），包含 50 条人类遥操作 demo。

```
{task_name}_demo.hdf5
├── data/                        # 顶层数据组
│   ├── attrs:
│   │   ├── env_name: "Libero_Tabletop_Manipulation"
│   │   ├── num_demos: 50
│   │   ├── total: 5068         # 所有 demo 总帧数
│   │   └── problem_info: {..., "language_instruction": "pick up the..."}
│   │
│   ├── demo_0/
│   │   ├── actions: (T, 7) float64       # 动作序列
│   │   ├── dones: (T,) uint8             # 是否完成
│   │   ├── rewards: (T,) float64         # 奖励
│   │   ├── states: (T, ...) float64      # MuJoCo 仿真器状态（用于精确重置）
│   │   ├── robot_states: (T, ...) float64 # 机器人完整状态
│   │   └── obs/                           # 观测
│   │       ├── agentview_rgb: (T, 128, 128, 3) uint8    # 第三人称相机
│   │       ├── eye_in_hand_rgb: (T, 128, 128, 3) uint8  # 手腕相机
│   │       ├── joint_states: (T, 7) float64              # 7个关节角度
│   │       ├── gripper_states: (T, 2) float64            # 左右夹爪开度
│   │       ├── ee_pos: (T, 3) float64                    # 末端执行器位置 xyz
│   │       ├── ee_ori: (T, 3) float64                    # 末端执行器姿态 rpy
│   │       └── ee_states: (T, 6) float64                 # ee_pos + ee_ori 拼接
│   │
│   ├── demo_1/
│   ├── ...
│   └── demo_49/
```

### Action Space（动作空间）

7 维连续动作，值域 [-1, 1]：

| 维度 | 含义 | 控制器映射 |
|------|------|-----------|
| 0-2 | 末端执行器位移 (dx, dy, dz) | OSC_POSE, output_max=[0.05, 0.05, 0.05] m |
| 3-5 | 末端执行器旋转 (droll, dpitch, dyaw) | OSC_POSE, output_max=[0.5, 0.5, 0.5] rad |
| 6 | 夹爪开合 | 1.0=闭合, -1.0=张开 |

控制器：OSC_POSE（操作空间阻抗控制），控制频率 20Hz。
action 值经过归一化：例如 dx=1.0 对应实际移动 0.05m。

### Observation Space（观测空间）

| 字段 | 维度 | 说明 |
|------|------|------|
| agentview_rgb | (128, 128, 3) | 固定第三人称俯视相机，RGB |
| eye_in_hand_rgb | (128, 128, 3) | 手腕安装相机，RGB |
| joint_states | (7,) | Panda 7个关节角度 (rad) |
| gripper_states | (2,) | 左右夹爪指尖位置 |
| ee_pos | (3,) | 末端执行器世界坐标 (m) |
| ee_ori | (3,) | 末端执行器欧拉角 (rad) |

注意：图像分辨率为 128×128（不是 224×224）。部分论文（如 FLOWER、OpenVLA）在评估时会 resize 到 224×224。

### 数据量统计（实际测量）

| Suite | 任务数 | Demo 数 | 总帧数 | 平均长度 | 标准差 | 最短 | 最长 |
|-------|--------|---------|--------|---------|--------|------|------|
| libero_spatial | 10 | 500 | 62,250 | 124 步 | 22 | 75 | 197 |
| libero_object | 10 | 500 | 74,507 | 149 步 | 20 | 114 | 254 |
| libero_goal | 10 | 500 | 63,728 | 127 步 | 42 | 75 | 347 |
| libero_10 | 10 | 500 | 138,090 | 276 步 | 63 | 150 | 517 |
| libero_90 | 90 | 4,500 | 669,043 | 149 步 | 45 | 58 | 373 |
| **全部合计** | **130** | **6,500** | **1,007,618** | **155 步** | - | 58 | 517 |

### 评估协议

**标准评估方式**：

1. 加载训练好的 policy
2. 对每个 task，从 LIBERO 提供的初始状态集合中选取状态
3. 用 policy rollout，最多 max_steps=**600** 步
4. 判断任务是否 success（LIBERO 用 BDDL 定义的 success condition 自动判断）
5. 每 task 跑 **20 rollouts**（快速验证）或 **50 rollouts**（论文最终）

**标准指标**：
- **Per-task success rate**：单个 task 成功率 = 成功 rollout 数 / 总 rollout 数
- **Per-suite average success**：suite 内 10 个 task 成功率的平均
- **Overall average**：所有 suite 的平均（论文主数字）

**FLOWER 的 LIBERO 评估**（参考 `flower_eval_libero.py`）：
- n_eval = 20（每 task 20 次 rollout）
- max_steps = 520
- multistep = 10（action chunking，每 10 步重新推理一次）
- 使用 LIBERO 的 `get_task_init_states()` 获取确定性初始状态

**CE-AIS 需要额外报告的指标**：
- intervention rate（CE-AIS 实际修改动作的比例）
- accepted / rejected / abstained rate
- action_delta_mean（动作平均修改量）
- energy_before / energy_after
- latency_ms（额外推理延迟）

### LIBERO vs CALVIN 对比

| 维度 | CALVIN | LIBERO |
|------|--------|--------|
| 机器人 | Franka Panda (7DOF) | Franka Panda (7DOF) |
| 动作维度 | 7 (6DOF + gripper) | 7 (6DOF + gripper) |
| 控制器 | 相对位移 | OSC_POSE 相对位移 |
| 图像分辨率 | 200×200 | 128×128 |
| 深度图 | 有 | 无 |
| 评估模式 | 5-task chain（连续5任务） | 单 task 独立 |
| 核心指标 | L1-L5, Avg Length | Per-task success rate |
| 长程压力 | 极高（错一步后面全失败） | 中等（单 task 独立） |
| Baseline 丰富度 | 中 | 高（OpenVLA, π0.5, DP 都有） |
| CE-AIS 价值 | 长程纠错 | model-agnostic 验证 |

### 论文中 LIBERO 各 suite 的定位

| Suite | 在论文中的作用 | CE-AIS 预期优势 |
|-------|---------------|----------------|
| libero_spatial | 验证空间理解不被 CE-AIS 破坏 | Clean preservation |
| libero_object | 验证物体识别不被干扰 | Clean preservation |
| libero_goal | 验证多目标任务兼容 | 部分任务可能有提升 |
| libero_10 | **主要展示 CE-AIS 价值** | 长程任务纠错、recovery |
| libero_90 | CE-WM 训练数据 / 泛化实验 | 证明 CE-WM 跨 task 泛化 |

### 其他论文如何使用 LIBERO（训练协议）

这是理解 LIBERO 的关键：**LIBERO 不是一个传统的 train/test split benchmark，而是每个 task 自带训练数据（50 条 demo），模型在这些 demo 上训练后在同一 task 上评估。** 这和 CALVIN ABC→D 的跨环境泛化设计完全不同。

#### 训练方式对比

| 方法 | 训练方式 | 训练数据 | 评估方式 | 备注 |
|------|---------|---------|---------|------|
| **π0.5 (OpenPI)** | 一个模型跑所有 suite | 4 suite 全部 demo 合并（1693 episodes） | 在各 suite 上分别评估 | 用 `physical-intelligence/libero` 数据集，包含 40 task 全部混合 |
| **OpenVLA-OFT** | 每个 suite 单独训一个模型 | 每次只用一个 suite 的 demo（10 task × 50 demo） | 在对应 suite 评估 | 也试过 4 suite 合并训一个，效果差不多 (97.1% vs 96.8%) |
| **FLOWER** | 每个 suite 单独训一个模型 | 每次只用一个 suite 的 demo | 在对应 suite 评估 | 也支持用 libero_90 预训练再 fine-tune |
| **LIBERO 官方 baseline** | 每个 suite 单独训一个模型 | 每次只用一个 suite 的 demo | 在对应 suite 评估 | BC-Transformer, Diffusion Policy |
| **MoDE** | 可预训练+微调 | libero_90 预训练 → libero_10 微调 | 在 libero_10 评估 | 类似 CALVIN 的预训练→迁移范式 |

#### 核心训练模式

**模式 1：Per-suite 独立训练（最常见）**
```
训练: libero_spatial 的 10 task × 50 demo = 500 条轨迹
评估: libero_spatial 的 10 task，每 task 50 rollouts
```
- OpenVLA-OFT、FLOWER、LIBERO 官方 baseline 都用这种方式
- 每个 suite 得到一个独立的 policy checkpoint
- 论文表格中报告每个 suite 的成绩

**模式 2：Multi-suite 联合训练（新趋势）**
```
训练: 4 suite 合并 = 40 task × 50 demo = 2000 条轨迹，混合在一起
评估: 在各 suite 上分别评估
```
- π0.5 用这种方式（一个模型跑所有 task）
- OpenVLA-OFT 也尝试过，报告为附录实验
- 通过 language instruction 区分 task（模型需要理解指令）

**模式 3：预训练 + 微调（迁移学习）**
```
预训练: libero_90 的 90 task（或 OXE 大规模数据）
微调: 在目标 suite 上微调
评估: 在目标 suite 上评估
```
- MoDE、FLOWER 支持这种方式
- 预训练数据和评估 task 不重叠
- 最接近 CALVIN ABC→D 的泛化精神

#### 关于"泛化性"的准确理解

你之前的质疑是对的。严格来说：

1. **LIBERO per-suite 评估不测跨环境泛化**——模型在 task A 的 demo 上训练，在 task A 上评估。"泛化"仅指对未见过的初始状态（微小扰动）的 robustness。

2. **LIBERO 测的其实是"学习效率"和"task 条件化能力"**——在只有 50 条 demo 的情况下，模型能否学会这个 task？10 个 task 放在一个 policy 里，模型能否通过 language 区分？

3. **真正的 held-out 泛化只有**：
   - libero_90 预训练 → 4 standard suite 评估（task 不重叠）
   - 4 suite 联合训练但 hold out 某些 task 评估（自己设计）

4. **对 CE-AIS 来说**，LIBERO 的价值不在于测 policy 泛化（policy 确实见过 demo），而在于：
   - 多个强 baseline 有公开数字，方便横向对比
   - 证明 CE-AIS 作为 test-time 插件对不同 backbone 都有效
   - libero_10 长程任务有足够的错误空间让 CE-AIS 发挥作用

#### OpenVLA-OFT 训练细节（参考）

```bash
# 训练 libero_spatial 的 OpenVLA-OFT
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --dataset_name libero_spatial_no_noops \  # 只用 spatial 的 demo
  --use_l1_regression True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 150005 \
  --lora_rank 32 \
  --image_aug True
```
- 训练 150K 步，8 GPU，约需半天
- LoRA fine-tuning（不是全量微调）
- 使用 RLDS 格式数据（`openvla/modified_libero_rlds`）
- `_no_noops` = 过滤掉了接近零的无效动作帧

#### π0.5 训练细节（参考）

```python
# pi05_libero config
data = LeRobotLiberoDataConfig(
    repo_id="physical-intelligence/libero",  # 4 suite 合并的数据
)
num_train_steps = 30_000  # 只需 30K 步（因为模型本身很强）
batch_size = 256
```
- 从 pi0.5 base 模型出发（已在大规模机器人数据上预训练）
- 只需 30K 步微调（因为 base 模型已经很强）
- 4 suite 混合训练，一个 checkpoint 评估所有 suite

---

### LIBERO 数据生态总览

除了原始官方 50 demo/task 数据，LIBERO 社区还有多种扩展和格式变体。以下是完整梳理。

#### 第一类：原始官方数据（已下载）

| 数据集 | 地址 | 内容 | 状态 |
|--------|------|------|------|
| `yifengzhu-hf/LIBERO-datasets` | https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets | 5 suite, 130 tasks, 6500 demos, ~100万帧 | ✅ 已下载到 `data/LIBERO-datasets/` |

#### 第二类：格式转换版（同一份数据，不同格式）

这些数据和原始数据内容完全相同，只是为不同框架做了格式转换。**不需要额外下载，对 CE-WM 无额外信息量。**

| 数据集 | 地址 | 格式 | 用途 |
|--------|------|------|------|
| `openvla/modified_libero_rlds` | https://huggingface.co/datasets/openvla/modified_libero_rlds | RLDS | OpenVLA/OpenVLA-OFT 训练用，4 suite ~10GB |
| `physical-intelligence/libero` | https://huggingface.co/datasets/physical-intelligence/libero | LeRobot v2 | π0/π0.5 训练用，4 suite 合并 |
| `nvidia/LIBERO_LeRobot_v3` | https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3 | LeRobot v3 | NVIDIA GR00T 等模型用 |
| `IPEC-COMMUNITY/libero_*_no_noops` | https://huggingface.co/IPEC-COMMUNITY | LeRobot | 过滤掉无效动作帧的版本 |
| `Ayana666888/LIBERO-3D` | https://huggingface.co/datasets/Ayana666888/LIBERO-3D | NPZ | 增加了深度图 + 点云 + 相机参数 |

#### 第三类：扩展数据集（新增数据，对 CE-AIS 有价值）

| 数据集 | 地址 | 内容 | 对 CE-AIS 的价值 |
|--------|------|------|-----------------|
| **LIBERO-X** (美团, RSS 2026) | https://huggingface.co/datasets/meituan/LIBERO-X | **600 tasks, 100 scenes, 2520 demos**。细粒度属性条件操作（颜色、纹理、大小）、空间推理 | **高**——大幅增加 CE-WM 训练数据多样性 |
| **SafeLIBERO** (清华) | https://huggingface.co/datasets/THURCSCT/SafeLIBERO | 4 suite × 4 task × 2 安全等级 × 50 episodes，带障碍物干扰 | **高**——天然的 OOD/safety 评估集，适合展示 CE-AIS 纠错能力 |

#### LIBERO-X 详情

- **论文**：https://arxiv.org/pdf/2602.06556
- **项目主页**：https://meituan.github.io/LIBERO-X/
- **数据集**：https://huggingface.co/datasets/meituan/LIBERO-X
- **规模**：600 tasks, 100 scenes, 2520 demonstrations
- **特点**：
  - 每个 scene 平均 6 个 task（原版 LIBERO 只有 2.6 个）
  - 属性条件操作（按颜色、大小、纹理区分物体）
  - 空间关系推理（左右、前后、远近）
  - 人类 VR 遥操作收集（Meta Quest 3）
- **格式**：LeRobot Parquet + MP4
- **对 CE-WM 训练的价值**：600 个 task 的多样性远超原版 130 个 task。如果用 LIBERO-X 训练 CE-WM，能覆盖更多操作模式，提升泛化能力。
- **下载**：
  ```bash
  huggingface-cli download meituan/LIBERO-X --repo-type dataset --local-dir data/LIBERO-X
  ```

#### SafeLIBERO 详情

- **项目主页**：https://vlsa-aegis.github.io/
- **数据集**：https://huggingface.co/datasets/THURCSCT/SafeLIBERO
- **规模**：4 suite × 4 task × 2 安全等级 × 50 episodes = 1600 evaluation episodes
- **特点**：
  - 在原版 LIBERO task 基础上添加障碍物（摩卡壶、储物箱、牛奶罐、酒瓶、杯子、书）
  - Level I：障碍物紧邻目标物体
  - Level II：障碍物阻挡运动路径
  - 物体和障碍物位置在每个 episode 中有小范围随机化
- **对 CE-AIS 的价值**：
  - 天然的 OOD 评估场景（训练时没有障碍物，测试时有）
  - 障碍物阻挡路径 → 原始 policy 可能失败 → CE-AIS 有机会纠正
  - 可以直接用作 Robustness Table 的实验
- **下载**：
  ```bash
  huggingface-cli download THURCSCT/SafeLIBERO --repo-type dataset --local-dir data/SafeLIBERO
  ```

#### CE-WM 训练数据选择总结

| 方案 | 数据源 | 估计帧数 | 建议 |
|------|--------|----------|------|
| 当前可用 | 原始 5 suite (已下载) | ~100 万帧 | ✅ 足够启动训练 |
| 扩展方案 | 原始 + LIBERO-X | ~100万 + LIBERO-X | 后续提升 CE-WM 泛化 |
| 评估扩展 | SafeLIBERO | 1600 episodes | 用于 OOD robustness 评估 |

**当前结论**：用已有的原始数据（libero_90 + 4 suite）训 CE-WM 完全够启动。LIBERO-X 和 SafeLIBERO 作为后续增强实验的数据源，建议在主实验跑通后再下载使用。

---

## 总览

```
Phase 1: 环境与数据准备（第1-2天）
  Step 1: LIBERO 环境安装与验证
  Step 2: LIBERO 数据集下载
  Step 3: FLOWER LIBERO checkpoint 下载与验证

Phase 2: Baseline 跑通（第3-5天）
  Step 4: FLOWER on LIBERO baseline 评估
  Step 5: OpenVLA-OFT checkpoint 下载与验证（可选并行）

Phase 3: CE-AIS 接入 LIBERO（第5-10天）
  Step 6: LIBERO 环境 wrapper 编写
  Step 7: LIBERO demo 数据 → CE-WM 训练数据转换
  Step 8: 在 LIBERO demo 上训练 encoder + CE-WM
  Step 9: CE-AIS + FLOWER LIBERO 联合评估

Phase 4: 多模型验证（第10-15天）
  Step 10: OpenVLA-OFT + CE-AIS
  Step 11: Diffusion Policy / MoDE baseline
  Step 12: π0.5 baseline（可选）
```

---

## 前置条件

- GPU 服务器，已安装 CUDA
- 当前项目 `/data0/yejinxuan/ce-ais` 已 clone 且 CALVIN 实验能跑
- 渲染支持：EGL 或 OSMesa（无头服务器用 `export MUJOCO_GL=egl`）

## 架构决策：LIBERO 独立安装

**问题**：LIBERO 原来嵌套在 `external/flower_vla_calvin/LIBERO/`（flower 仓库自带），但 CE-AIS 需要在多个模型上使用 LIBERO（FLOWER、OpenVLA-OFT、MoDE、π0.5 等）。如果依赖 flower 内的路径，其他模型需要手动 `sys.path.insert`，维护成本高。

**解决方案**：将 LIBERO 独立 clone 到 `external/LIBERO/` 并 `pip install -e`，注册为全局 Python 包。

**目录结构**：
```
external/
├── LIBERO/                      ← 独立安装，所有模型共用（主 LIBERO）
├── flower_vla_calvin/
│   ├── LIBERO/                  ← 仍存在但不再作为主包使用
│   ├── flower/                  ← FLOWER 模型代码
│   └── ...
├── openvla-oft/                 ← 后续添加
├── MoDE_Diffusion_Policy/       ← 后续添加
└── openpi/                      ← 后续添加（可选）
```

**已完成的操作**：
1. `git clone --depth 1` LIBERO 到 `external/LIBERO/`
2. `pip install -e . --config-settings editable_mode=compat`
3. 创建 `~/.libero/config.yaml` 指定数据路径为 `/data0/yejinxuan/ce-ais/data/libero_datasets`
4. 验证 `import libero.libero` 指向新路径

---

## Phase 1: 环境与数据准备

### Step 1: LIBERO 环境安装与验证

**架构决策**：LIBERO 作为独立仓库 clone 到 `external/LIBERO/`（而非使用 flower submodule 内的嵌套副本）。原因：CE-AIS 需要在多个底层模型（FLOWER、OpenVLA-OFT、MoDE、π0.5）上使用 LIBERO，独立安装后所有模型统一通过 `from libero.libero import ...` 访问，不依赖 flower 路径。

> 注意：`external/flower_vla_calvin/LIBERO/` 仍存在但不再作为主 LIBERO 包使用。所有新代码应 import 独立安装的版本。

**1.1 Clone LIBERO 到独立位置（已完成）**

```bash
cd /data0/yejinxuan/ce-ais/external
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

当前位置：`/data0/yejinxuan/ce-ais/external/LIBERO/`

**1.2 安装为 editable 包（已完成）**

```bash
cd /data0/yejinxuan/ce-ais/external/LIBERO
pip install -e . --config-settings editable_mode=compat
```

注意：Python 3.13 下必须用 `--config-settings editable_mode=compat`，否则 setuptools 的 finder 机制会导致 MAPPING 为空。

**1.3 配置 LIBERO 路径（已完成）**

首次 import LIBERO 会交互式询问数据路径。为避免这个问题，直接创建配置文件：

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /data0/yejinxuan/ce-ais/external/LIBERO/libero/libero
bddl_files: /data0/yejinxuan/ce-ais/external/LIBERO/libero/libero/bddl_files
init_states: /data0/yejinxuan/ce-ais/external/LIBERO/libero/libero/init_files
datasets: /data0/yejinxuan/ce-ais/data/libero_datasets
assets: /data0/yejinxuan/ce-ais/external/LIBERO/libero/libero/assets
EOF
```

配置文件位置：`~/.libero/config.yaml`（已创建）

**1.4 安装 LIBERO 运行时依赖**

LIBERO 的 `benchmark` 模块需要 torch，`envs` 模块需要 robosuite + mujoco：

```bash
# 核心依赖（根据 GPU 环境选择 torch 版本）
pip install torch torchvision  # 如果环境中还没有
pip install robosuite==1.4.0
pip install mujoco
pip install hydra-core omegaconf
pip install h5py opencv-python termcolor
```

**1.5 验证 LIBERO 能正常 import**

```python
python -c "
from libero.libero import benchmark, get_libero_path
print('LIBERO benchmark suites:', list(benchmark.get_benchmark_dict().keys()))
print('BDDL path:', get_libero_path('bddl_files'))
print('Init states path:', get_libero_path('init_states'))
import libero.libero
print('Import source:', libero.libero.__file__)
# 应指向: /data0/yejinxuan/ce-ais/external/LIBERO/libero/libero/__init__.py
"
```

验证：
- 输出包含 `libero_spatial`, `libero_object`, `libero_goal`, `LIBERO_10`, `LIBERO_90`
- Import source 指向 `external/LIBERO/`（不是 `external/flower_vla_calvin/LIBERO/`）

**1.6 验证渲染**

```bash
export MUJOCO_GL=egl  # 或 osmesa（无头服务器）
python -c "
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path
import os

bm = benchmark.get_benchmark_dict()['libero_spatial']()
task = bm.get_task(0)
bddl_folder = get_libero_path('bddl_files')

env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(bddl_folder, task.problem_folder, task.bddl_file),
    camera_heights=224,
    camera_widths=224,
)
obs = env.reset()
print('Obs keys:', list(obs.keys()))
print('agentview_image shape:', obs['agentview_image'].shape)
env.close()
print('LIBERO rendering OK!')
"
```

验证：应输出 `agentview_image shape: (224, 224, 3)` 且无报错。

---

### Step 2: LIBERO 数据集下载

LIBERO 数据集包含每个 suite 的 demo trajectories（每 task 50 条 demo，HDF5 格式）。

**2.1 下载地址（三选一）**

方式 A：用 LIBERO 自带脚本（推荐）
```bash
cd /data0/yejinxuan/ce-ais/external/LIBERO
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```
这会下载所有 suite 到 `~/.libero/datasets/` 或配置文件中指定的路径（当前为 `/data0/yejinxuan/ce-ais/data/libero_datasets`）。

方式 B：HuggingFace 手动下载
- 地址：https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
```bash
pip install huggingface_hub
huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset --local-dir /data0/yejinxuan/ce-ais/data/LIBERO-datasets
```

当前实际位置：`/data0/yejinxuan/ce-ais/data/LIBERO-datasets`（已下载完成）

方式 C：UT Austin Box 直链下载（备用，可能过期）
```bash
mkdir -p /data0/yejinxuan/ce-ais/data/libero_datasets && cd $_
wget -O libero_spatial.zip "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
wget -O libero_object.zip "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip"
wget -O libero_goal.zip "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip"
wget -O libero_100.zip "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip"
unzip "*.zip"
```

**2.2 数据集内容**

| Suite | 文件 | 内容 |
|-------|------|------|
| libero_spatial | 10 个 .hdf5 | 每 task 50 条 demo, 空间关系任务 |
| libero_object | 10 个 .hdf5 | 每 task 50 条 demo, 物体操作任务 |
| libero_goal | 10 个 .hdf5 | 每 task 50 条 demo, 目标条件任务 |
| libero_100 | libero_90 + libero_10 | 90 task 预训练 + 10 task 长程评估 |

**2.3 验证数据集**

```python
import h5py, glob

# 找到下载的数据目录
data_dir = "/data0/yejinxuan/ce-ais/data/libero_datasets"  # 根据实际路径调整
files = glob.glob(f"{data_dir}/libero_spatial/*.hdf5")
print(f"libero_spatial files: {len(files)}")  # 应为 10

f = h5py.File(files[0], 'r')
print(f"Keys: {list(f.keys())}")
print(f"Demo count: {len(list(f['data'].keys()))}")  # 应为 50
demo = f['data/demo_0']
print(f"Demo keys: {list(demo.keys())}")
print(f"Actions shape: {demo['actions'].shape}")  # (T, 7)
print(f"Obs keys: {list(demo['obs'].keys())}")
f.close()
```

验证：每个 suite 有 10 个 .hdf5 文件，每个文件有 50 条 demo，action dim=7。

---

### Step 3: FLOWER LIBERO checkpoint 下载与验证

FLOWER 有针对每个 LIBERO suite 的预训练 checkpoint。

**3.1 下载地址**

HuggingFace 模型列表：
| Suite | HuggingFace Model ID | 下载命令 |
|-------|---------------------|----------|
| Spatial | `mbreuss/flower_libero_spatial` | `huggingface-cli download mbreuss/flower_libero_spatial --local-dir checkpoints/flower_libero_spatial` |
| Object | `mbreuss/flower_libero_object` | `huggingface-cli download mbreuss/flower_libero_object --local-dir checkpoints/flower_libero_object` |
| Goal | `mbreuss/flower_libero_goal` | `huggingface-cli download mbreuss/flower_libero_goal --local-dir checkpoints/flower_libero_goal` |
| LIBERO-10 | `mbreuss/flower_libero_10` | `huggingface-cli download mbreuss/flower_libero_10 --local-dir checkpoints/flower_libero_10` |
| LIBERO-90 | `mbreuss/flower_libero_90` | `huggingface-cli download mbreuss/flower_libero_90 --local-dir checkpoints/flower_libero_90` |
| Pretrained base | `mbreuss/flower_vla_pret` | `huggingface-cli download mbreuss/flower_vla_pret --local-dir checkpoints/flower_vla_pret` |

**建议优先下载**：`flower_libero_spatial`（最快验证）和 `flower_libero_10`（长程任务，最适合 CE-AIS）。

```bash
cd /data0/yejinxuan/ce-ais
pip install huggingface_hub

# 优先下载这两个
huggingface-cli download mbreuss/flower_libero_spatial --local-dir checkpoints/flower_libero_spatial
huggingface-cli download mbreuss/flower_libero_10 --local-dir checkpoints/flower_libero_10
```

**3.2 验证 checkpoint 文件**

```bash
ls checkpoints/flower_libero_spatial/
# 应看到 .ckpt 或 .safetensors 文件, 以及 config.yaml / .hydra/ 等
```

```python
import torch
ckpt_path = "checkpoints/flower_libero_spatial/"  # 找到 .ckpt 文件
import glob
ckpts = glob.glob(f"{ckpt_path}/**/*.ckpt", recursive=True)
print(f"Found checkpoints: {ckpts}")
# 尝试加载确认不报错
state = torch.load(ckpts[0], map_location='cpu')
print(f"Keys: {list(state.keys())[:10]}")
```

验证：checkpoint 能被 torch.load 正常加载，有 `state_dict` 或模型权重 keys。

---

## Phase 2: Baseline 跑通

### Step 4: FLOWER on LIBERO baseline 评估

目标：用 FLOWER 官方评估脚本在 LIBERO 上跑通 frozen baseline，确认能复现论文数字。

**4.1 评估脚本位置**

已存在：`external/flower_vla_calvin/flower/evaluation/flower_eval_libero.py`

**4.2 配置文件准备**

修改 `external/flower_vla_calvin/conf/eval_libero.yaml`，将路径改为本地：

```yaml
# 需要修改的关键字段：
log_dir: /data0/yejinxuan/ce-ais/logs/libero_eval/
checkpoint: /data0/yejinxuan/ce-ais/checkpoints/flower_libero_spatial/xxx.ckpt  # 实际 ckpt 路径
train_folder: /data0/yejinxuan/ce-ais/checkpoints/flower_libero_spatial/.hydra/config.yaml  # hydra config
benchmark_name: libero_spatial
n_eval: 20  # 每 task 20 rollouts（先用少量验证）
device: 0
```

**4.3 运行评估**

```bash
cd /data0/yejinxuan/ce-ais/external/flower_vla_calvin
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python flower/evaluation/flower_eval_libero.py
```

**4.4 验证标准**

FLOWER 论文报告的 LIBERO 成绩（参考值）：
| Suite | FLOWER 论文 Success Rate |
|-------|------------------------|
| libero_spatial | ~96-98% |
| libero_object | ~96-98% |
| libero_goal | ~96-98% |
| libero_10 | ~90-95% |

如果本地跑出来 success rate 接近上述数字（±3%），则 baseline 验证通过。

---

### Step 5: OpenVLA-OFT checkpoint 下载（可与 Step 4 并行）

OpenVLA-OFT 是 LIBERO 上最重要的 VLA baseline，有针对每个 suite 的 fine-tuned checkpoint。

**5.1 下载 OpenVLA-OFT 代码**

```bash
cd /data0/yejinxuan/ce-ais/external
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .
```

**5.2 下载 checkpoint**

HuggingFace 模型（每个约 14GB）：
| Suite | Model ID |
|-------|----------|
| Spatial | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| Object | `moojink/openvla-7b-oft-finetuned-libero-object` |
| Goal | `moojink/openvla-7b-oft-finetuned-libero-goal` |
| LIBERO-10 | `moojink/openvla-7b-oft-finetuned-libero-10` |
| Combined (all suites) | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |

```bash
# 先下载 Spatial 验证
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir /data0/yejinxuan/ce-ais/checkpoints/openvla_oft_libero_spatial
```

注意：OpenVLA-OFT 是 7B 参数模型，需要 A100 40GB 或同等 GPU。支持 4-bit/8-bit 量化推理。

**5.3 验证**

```bash
cd /data0/yejinxuan/ce-ais/external/openvla-oft
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5
```

验证：OpenVLA-OFT 论文报告 libero_spatial ~96%+ 成功率。

---

## Phase 3: CE-AIS 接入 LIBERO

### Step 6: 编写 LIBERO 环境 wrapper

目标：创建 `src/evaluation/libero_integration.py`，与现有 `calvin_integration.py` 接口对齐。

**6.1 需要实现的核心类**

```python
class LIBEROWrapper:
    """封装 LIBERO 环境，统一为 CE-AIS 可用的接口"""

    def __init__(self, benchmark_name, task_idx, img_size=224):
        ...

    def reset(self, init_state=None):
        """返回 observation dict: rgb_static, rgb_gripper, robot_obs"""
        ...

    def step(self, action):
        """执行 action (7-dim), 返回 obs, reward, done, info"""
        ...

    def get_obs_for_encoder(self, obs):
        """将 LIBERO obs 转为 CE-AIS encoder 输入格式"""
        # rgb_static ← agentview_image (224,224,3)
        # rgb_gripper ← robot0_eye_in_hand_image (224,224,3)
        # robot_obs ← robot0_joint_pos + robot0_gripper_qpos
        ...
```

**6.2 关键 obs 映射**

| LIBERO 字段 | CE-AIS 字段 | 维度 |
|-------------|-------------|------|
| `agentview_image` | `rgb_static` | (224,224,3) |
| `robot0_eye_in_hand_image` | `rgb_gripper` | (224,224,3) |
| `robot0_joint_pos` | `robot_obs[:7]` | (7,) |
| `robot0_gripper_qpos` | `robot_obs[7:9]` | (2,) |

**6.3 验证**

```python
from src.evaluation.libero_integration import LIBEROWrapper
env = LIBEROWrapper("libero_spatial", task_idx=0)
obs = env.reset()
assert obs['rgb_static'].shape == (224, 224, 3)
assert obs['robot_obs'].shape[0] >= 7
action = np.zeros(7)
obs, reward, done, info = env.step(action)
print("LIBEROWrapper OK")
```

---

### Step 7: LIBERO demo 数据 → CE-WM 训练数据转换

CE-WM 需要 `(z_seq, a_seq)` 格式的训练数据。需要从 LIBERO HDF5 demo 中提取。

**7.1 数据加载器（已完成）**

已创建 `src/data/libero_dataset.py`，与 `src/data/calvin_dataset.py` 接口对齐。

LIBERO HDF5 数据结构（实际验证）：
```
data/
  demo_0/
    actions: (T, 7) float64, range [-1, 1]
    obs/
      agentview_rgb: (T, 128, 128, 3) uint8
      eye_in_hand_rgb: (T, 128, 128, 3) uint8
      joint_states: (T, 7) float64
      gripper_states: (T, 2) float64
      ee_pos: (T, 3) float64
      ee_ori: (T, 3) float64
      ee_states: (T, 6) float64
  demo_1/
  ...
  demo_49/
```

字段映射：
| LIBERO 字段 | CE-AIS 字段 | 维度 | 说明 |
|-------------|-------------|------|------|
| `agentview_rgb` | `rgb` | (128,128,3)→(3,128,128) | ImageNet 归一化 |
| `joint_states` + `gripper_states` | `pose` | (9,) | 7 关节角 + 2 夹爪 |
| `actions` | `a_pos` | (7,) | 已归一化到 [-1,1] |

使用方式：
```python
from src.data.libero_dataset import LIBERODataset

# 单 suite 快速验证
ds = LIBERODataset('data/LIBERO-datasets', suite_names='libero_spatial', mode='ce_wm')

# 4 suite 合并（推荐）
ds = LIBERODataset(
    'data/LIBERO-datasets',
    suite_names='libero_spatial,libero_object,libero_goal,libero_10',
    mode='ce_wm',
)

# encoder 模式
ds_enc = LIBERODataset('data/LIBERO-datasets', suite_names='libero_spatial', mode='encoder')

# 查看统计
print(ds.get_stats())
```

注意事项：
- LIBERO 图像为 128×128（CALVIN 是 200×200），`image_size` 默认 (128,128) 不做 resize
- LIBERO 没有深度图（CALVIN 有 `depth_static`），`__getitem__` 输出不含 `depth_seq`
- `pose_dim=9`（CALVIN 是 7），因为 LIBERO 有 7 关节角 + 2 夹爪状态
- 数据全部加载到内存（4 suite 合并约需 ~4GB RAM）

**7.2 数据量分析：LIBERO vs CALVIN 对比（已验证实际数据）**

CE-WM 在 CALVIN 上的训练规模：
- CALVIN ABC 三环境连续轨迹：**~179 万帧**
- 滑窗 window_size=16，产生 ~179 万个训练 window
- 训练 200 epoch，总计暴露 ~5.7 亿 frame-tokens

LIBERO 实际数据量（已用 `LIBERODataset` 验证）：

| 训练方案 | 数据源 | 实际帧数 | CE-WM windows | 对比 CALVIN |
|---------|--------|----------|--------------|------------|
| A: 单 suite (spatial) | 10 task × 50 demo | 62,250 | 54,750 | CALVIN 的 3% |
| B: 4 suite 合并 | 40 task × 50 demo | **338,575** | **308,575** | CALVIN 的 17% |
| C: LIBERO-90 | 90 task × 50 demo | ~65 万 (估) | ~58 万 (估) | CALVIN 的 32% |
| D: 全部 (90 + 4 suite) | 130 task × 50 demo | ~99 万 (估) | ~89 万 (估) | CALVIN 的 50% |

各 suite 实际统计：
| Suite | Demos | 总帧数 | 平均 demo 长度 | 最短 | 最长 |
|-------|-------|--------|---------------|------|------|
| libero_spatial | 500 | 62,250 | 124 步 | 75 | 197 |
| libero_object | 500 | 74,507 | 149 步 | - | - |
| libero_goal | 500 | 63,728 | 127 步 | - | - |
| libero_10 | 500 | 138,090 | 276 步 | - | - |
| libero_90 | ~4500 | ~65 万 | ~146 步 | - | - |

**结论：LIBERO demo 比预估的短很多（平均 124-276 步，不是 300-500 步）。4 suite 合并只有 33.8 万帧，远少于 CALVIN。**

应对策略：
- **增加 epoch 数**：4 suite 合并时用 400-600 epoch（而非 CALVIN 的 200）来补偿数据量
- **减小 window_size**：可以尝试 window_size=8（产生更多训练样本）
- **加入 LIBERO-90**：方案 D 能达到 ~99 万帧，更接近合理规模
- **CE-WM 是判别模型**：数据效率本身就比生成模型高，17% 的数据量不一定意味着 17% 的性能

为什么合并 suite 训练可行：
1. CE-WM 学的是"状态-动作对是否合理"的能量判别，不绑定具体 task
2. LIBERO 所有 suite 共享同一个机器人、同一个 action space (7-dim)
3. 多 task 训练能让 CE-WM 学到更通用的动力学先验
4. 在论文中，"跨 task 泛化的 CE-WM" 本身也是一个有价值的实验点

**额外数据源（可选）**：
- `openvla/modified_libero_rlds`（HuggingFace, ~10GB）：OpenVLA 用的 LIBERO RLDS 格式数据
- `physical-intelligence/libero`（HuggingFace）：PI 整理的版本
- 如果需要更多 demo，可以用已训练好的 FLOWER/OpenVLA-OFT 在 LIBERO 环境中 rollout 收集 successful trajectories 作为额外正样本

**7.3 验证（已通过）**

```
=== Single Suite (libero_spatial, ce_wm mode) ===
  n_trajectories: 500
  total_frames: 62250
  n_samples: 54750
  avg_traj_length: 124.5
  mode: ce_wm
  window_size: 16

Sample: rgb_seq=(16, 3, 128, 128), pose_seq=(16, 9), a_pos=(16, 7)
Action range: [-1.000, 0.876]

=== 4-Suite Combined (ce_wm mode) ===
  n_trajectories: 2000
  total_frames: 338575
  n_samples: 308575

=== Encoder mode ===
  n_samples: 61750
  rgb=(3, 128, 128), pose=(9,)
```

---

### Step 8: 在 LIBERO demo 上训练 encoder + CE-WM

**8.1 训练 encoder（对比学习）**

```bash
cd /data0/yejinxuan/ce-ais
python scripts/pretrain.py \
  --stage encoder \
  --dataset libero \
  --data-dir data/libero_datasets \
  --suite libero_spatial \
  --epochs 30 \
  --batch-size 512 \
  --device cuda:0
```

验证：loss 下降到 < 1.0，encoder 能将不同帧映射为有区分度的 embedding。

**8.2 训练 CE-WM（能量模型）**

```bash
python scripts/pretrain.py \
  --stage ce_wm \
  --dataset libero \
  --data-dir data/libero_datasets \
  --suite libero_spatial \
  --encoder-ckpt checkpoints/libero_encoder_best.pt \
  --epochs 200 \
  --batch-size 256 \
  --device cuda:0
```

验证：
- `pos_energy_mean` < `neg_energy_mean`（正样本能量低于负样本）
- energy margin > 0 且稳定
- loss 收敛

**8.3 关键决策：训练一个通用 CE-WM 还是 per-suite CE-WM？**

基于上面 Step 7.2 的数据量分析，具体建议如下：

**阶段 1（验证 pipeline）：Per-suite CE-WM**
```bash
# 只用 libero_spatial 的 500 条 demo (~15万帧) 快速验证
python scripts/pretrain.py \
  --stage ce_wm \
  --dataset libero \
  --data-dir data/libero_datasets \
  --suite libero_spatial \
  --epochs 400 \  # 数据少，需要更多 epoch 补偿
  --batch-size 128 \  # 相应减小 batch
  --device cuda:0
```

**阶段 2（论文主实验）：Cross-suite CE-WM**
```bash
# 合并 4 suite (2000 条 demo, ~60-100万帧)
python scripts/pretrain.py \
  --stage ce_wm \
  --dataset libero \
  --data-dir data/libero_datasets \
  --suite libero_spatial,libero_object,libero_goal,libero_10 \
  --epochs 200 \
  --batch-size 256 \
  --device cuda:0
```

**阶段 3（泛化实验）：LIBERO-90 CE-WM → 评估 standard suites**
```bash
# 用 90 task 训练 (~135-225万帧, 与 CALVIN 规模相当)
python scripts/pretrain.py \
  --stage ce_wm \
  --dataset libero \
  --data-dir data/libero_datasets \
  --suite libero_90 \
  --epochs 200 \
  --batch-size 256 \
  --device cuda:0
# 然后在 spatial/object/goal/libero_10 上评估 → 证明 CE-WM 泛化能力
```

**训练超参调整建议（相对 CALVIN）：**

| 参数 | CALVIN | LIBERO 单 suite | LIBERO 4-suite |
|------|--------|----------------|----------------|
| 数据量 | 179 万帧 | ~15-25 万帧 | ~60-100 万帧 |
| batch_size | 256 | 128 | 256 |
| epochs | 200 | 300-400 | 200 |
| window_size | 16 | 16（不变） | 16（不变） |
| learning_rate | 原值 | 适当降低(×0.5) | 原值 |
| 数据增强 | 无 | 考虑 action noise augmentation | 可选 |

**论文表格设计**：可以做一组"CE-WM 训练数据规模消融"：

| CE-WM Training Data | Spatial Success | Object Success | Goal Success | Long Success |
|---------------------|----------------|----------------|--------------|--------------|
| Per-suite (单 suite) | | | | |
| Cross-suite (4 suite) | | | | |
| LIBERO-90 | | | | |

---

### Step 9: CE-AIS + FLOWER LIBERO 联合评估

**9.1 创建评估脚本**

创建 `scripts/run_libero_experiments.py`，参考 `scripts/run_paper_experiments.py` 结构：

```bash
python scripts/run_libero_experiments.py \
  --suite libero_spatial \
  --methods frozen_flower ce_ais \
  --flower-ckpt checkpoints/flower_libero_spatial/ \
  --cewm-ckpt checkpoints/libero_cewm_spatial.pt \
  --encoder-ckpt checkpoints/libero_encoder_spatial.pt \
  --n-eval 20 \
  --device cuda:0
```

**9.2 输出格式**

与 CALVIN 实验对齐的 JSON：
```json
{
  "suite": "libero_spatial",
  "methods": {
    "frozen_flower": {
      "per_task_success": [0.95, 0.90, ...],  // 10 tasks
      "avg_success": 0.93,
      "latency_ms": 45
    },
    "ce_ais": {
      "per_task_success": [0.95, 0.95, ...],
      "avg_success": 0.95,
      "latency_ms": 250,
      "diagnostics": {
        "accepted_rate": 0.35,
        "rejected_rate": 0.15,
        "energy_before_mean": 0.12,
        "energy_after_mean": 0.08,
        "action_delta_mean": 0.03
      }
    }
  }
}
```

**9.3 验证标准**

- CE-AIS 在 libero_spatial 上 ≥ frozen FLOWER（不破坏 clean performance）
- CE-AIS 在 libero_10（长程任务）上有可见提升
- Diagnostics 显示合理的 intervention rate（不是 0% 也不是 100%）

---

## Phase 4: 多模型验证

### Step 10: OpenVLA-OFT + CE-AIS

**10.1 适配 OpenVLA-OFT 到 VLAAdapter**

你已有 `src/dual_stream/vla_adapter.py` 中的 `OpenVLAAdapter`。需要确认：
- LIBERO 的 image size (224×224) 与 OpenVLA 输入匹配（OpenVLA 用 224×224）
- Action space 一致（7-dim: 6 DOF + gripper）
- 确认 `--center_crop True` 参数（OpenVLA-OFT 训练时用了 random crop augmentation）

**10.2 运行**

```bash
python scripts/run_libero_experiments.py \
  --suite libero_spatial \
  --methods frozen_openvla_oft ce_ais_openvla_oft \
  --openvla-ckpt moojink/openvla-7b-oft-finetuned-libero-spatial \
  --cewm-ckpt checkpoints/libero_cewm_spatial.pt \
  --encoder-ckpt checkpoints/libero_encoder_spatial.pt \
  --n-eval 20 \
  --device cuda:0
```

**10.3 验证**

OpenVLA-OFT 论文数字：
| Suite | OpenVLA-OFT Success |
|-------|-------------------|
| Spatial | 96.0% |
| Object | 92.0% |
| Goal | 84.0% |
| LIBERO-10 | 80.0% |

如果 CE-AIS 能在 Goal 和 LIBERO-10 上有提升（这两个 baseline 相对较弱），即为正面结果。

---

### Step 11: Diffusion Policy / MoDE baseline

**11.1 方案选择**

推荐使用 MoDE（Mixture of Expert Denoisers），因为它是 FLOWER 同组的 diffusion 方法，有 LIBERO checkpoint：

| 模型 | HuggingFace ID |
|------|---------------|
| MoDE LIBERO-10 | `mbreuss/MoDE_LIBERO_10` |
| MoDE pretrained | `mbreuss/MoDE_pret` |

MoDE 代码仓库：https://github.com/intuitive-robots/MoDE_Diffusion_Policy

```bash
# 下载 MoDE
cd /data0/yejinxuan/ce-ais/external
git clone https://github.com/intuitive-robots/MoDE_Diffusion_Policy.git
cd MoDE_Diffusion_Policy && pip install -e .

# 下载 checkpoint
huggingface-cli download mbreuss/MoDE_LIBERO_10 --local-dir /data0/yejinxuan/ce-ais/checkpoints/mode_libero_10
```

**11.2 备选：LIBERO 内置 baseline**

LIBERO 官方 repo 自带 BC-Transformer 和 Diffusion Policy 训练：
```bash
cd /data0/yejinxuan/ce-ais/external/flower_vla_calvin/LIBERO
python libero/lifelong/main.py \
  --benchmark_name libero_spatial \
  --algo_name bc_transformer \
  --seed 42
```

这个更轻量但需要自己训练。

**11.3 CE-AIS 接入 Diffusion Policy 的关键**

Diffusion Policy 输出的是 action chunk（多步动作序列），而不是单步动作。CE-AIS 可以：
- 对 chunk 中的每一步分别做能量验证
- 或对整个 chunk 做能量评估后 accept/reject

建议先用"逐步验证"方式，与 FLOWER（multistep=10）的处理模式一致。

---

### Step 12: π0.5 baseline（可选，高影响力）

**12.1 下载 OpenPI**

```bash
cd /data0/yejinxuan/ce-ais/external
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git submodule update --init --recursive
```

**12.2 π0.5 LIBERO checkpoint**

存储在 Google Cloud Storage：
```
gs://openpi-assets/checkpoints/pi05_libero/
```

下载：
```bash
# 需要 gsutil
pip install google-cloud-storage
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/ /data0/yejinxuan/ce-ais/checkpoints/pi05_libero/
```

或者查看 HuggingFace 社区镜像：
- `Tacoin/openpi-pi0.5-libero-onnx`

**12.3 运行方式**

OpenPI 推荐 Docker 方式运行（server-client 架构）：
```bash
cd /data0/yejinxuan/ce-ais/external/openpi
# Server
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# Client (另一个 terminal)
CLIENT_ARGS="--args.task-suite-name libero_spatial" docker compose -f examples/libero/compose.yml run client
```

也可以不用 Docker，直接用 Python API：
```python
from openpi.policies import load_policy
policy = load_policy("pi05_libero", checkpoint_dir="checkpoints/pi05_libero")
action = policy.predict(observation)
```

**12.4 验证**

π0.5 论文报告 LIBERO 4 suite 平均 ~96.85% 成功率，是目前最强 baseline 之一。

**12.5 注意事项**

- π0.5 需要较大显存（建议 A100 80GB）
- License 需确认是否允许学术使用
- 如果接入困难，可作为 Tier 2 推后，先专注 FLOWER + OpenVLA-OFT

---

## 下载资源汇总表

| 资源 | 地址 | 大小（约） | 优先级 |
|------|------|-----------|--------|
| LIBERO 数据集 | HuggingFace `yifengzhu-hf/LIBERO-datasets` | ~10GB | 必须 |
| FLOWER libero_spatial ckpt | HuggingFace `mbreuss/flower_libero_spatial` | ~2-5GB | 必须 |
| FLOWER libero_10 ckpt | HuggingFace `mbreuss/flower_libero_10` | ~2-5GB | 必须 |
| FLOWER libero_object ckpt | HuggingFace `mbreuss/flower_libero_object` | ~2-5GB | 高 |
| FLOWER libero_goal ckpt | HuggingFace `mbreuss/flower_libero_goal` | ~2-5GB | 高 |
| OpenVLA-OFT spatial ckpt | HuggingFace `moojink/openvla-7b-oft-finetuned-libero-spatial` | ~14GB | 高 |
| OpenVLA-OFT combined ckpt | HuggingFace `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | ~14GB | 高 |
| MoDE LIBERO-10 | HuggingFace `mbreuss/MoDE_LIBERO_10` | ~2GB | 中 |
| π0.5 LIBERO | `gs://openpi-assets/checkpoints/pi05_libero/` | ~10-20GB | 可选 |
| OpenPI 代码 | GitHub `Physical-Intelligence/openpi` | - | 可选 |
| OpenVLA-OFT 代码 | GitHub `moojink/openvla-oft` | - | 高 |
| MoDE 代码 | GitHub `intuitive-robots/MoDE_Diffusion_Policy` | - | 中 |

---

## 需要新写的代码文件清单

| 文件 | 作用 | 参考 |
|------|------|------|
| `src/evaluation/libero_integration.py` | LIBERO env wrapper | `src/evaluation/calvin_integration.py` |
| `src/data/libero_dataset.py` | LIBERO demo 数据加载 | `src/data/calvin_dataset.py` |
| `scripts/run_libero_experiments.py` | LIBERO 主评估脚本 | `scripts/run_paper_experiments.py` |
| `src/dual_stream/vla_adapter.py` 新增 | `FlowerLIBEROAdapter` | 现有 `FlowerVLAAdapter` |
| `configs/libero_base.yaml` | LIBERO 实验配置 | `configs/base.yaml` |

---

## 常见问题

**Q: CE-WM 应该在哪个数据上训练？**

A: 在 LIBERO 的 demo 轨迹上训练（和 CALVIN 上的做法一致：expert demo → encoder embedding → CE-WM）。建议先在单个 suite（如 libero_spatial）上验证 pipeline，再扩展到多 suite。

**Q: LIBERO 和 CALVIN 的 action space 一样吗？**

A: 基本一致，都是 7-dim（6 DOF + gripper）。但具体的 action scale 和 normalization 可能不同。需要在 adapter 中处理。

**Q: 如果 FLOWER LIBERO checkpoint 跑不通怎么办？**

A: 直接用 LIBERO 内置的 BC-Transformer 训练一个 baseline（几小时即可），作为 CE-AIS 的底层策略。这样也能验证 CE-AIS 的 model-agnostic 特性。

**Q: 是否需要为每个 baseline 单独训练 CE-WM？**

A: 不需要。CE-WM 评价的是"状态-动作对是否合理"，与哪个 policy 产生这个动作无关。一个在 expert demo 上训练好的 CE-WM 可以用于评估所有 baseline 的动作。

---

## 里程碑检查点

- [x] LIBERO 独立 clone 到 `external/LIBERO/` 并 pip install（已完成）
- [x] `~/.libero/config.yaml` 配置创建（已完成，datasets 指向 `data/LIBERO-datasets`）
- [x] `import libero.libero` 路径指向 `external/LIBERO/`（已验证）
- [x] LIBERO 数据集下载完毕（已在 `data/LIBERO-datasets/`，包含 5 个 suite）
- [x] `src/data/libero_dataset.py` 编写完成并测试通过（已验证 encoder + ce_wm 模式）
- [ ] LIBERO 运行时依赖安装（robosuite, mujoco 渲染）
- [ ] LIBERO 环境能渲染（Step 1.6 完成）
- [ ] FLOWER LIBERO checkpoint 下载并加载成功（Step 3 完成）
- [ ] FLOWER on LIBERO baseline 跑通，成功率合理（Step 4 完成）
- [ ] `libero_integration.py` 编写完成并测试（Step 6 完成）
- [ ] CE-WM 在 LIBERO demo 上训练完成（Step 8 完成）
- [ ] CE-AIS + FLOWER LIBERO 联合评估有结果（Step 9 完成）
- [ ] OpenVLA-OFT baseline 跑通（Step 10 完成）
- [ ] 至少两个 baseline + CE-AIS 的 LIBERO 主表格填满（Phase 4 完成）
