# CE-AIS: Causal-Energy Active Inference Steering

无梯度测试时自适应框架，用于具身智能 VLA 策略的在线校正。

核心思路：冻结 VLA 参数，通过因果能量世界模型 (CE-WM) 在推理时评估候选动作的能量，利用退火朗之万动力学偏转动作轨迹，实现无需微调的 OOD 鲁棒性提升。

## 项目结构

```
ce-ais/
├── configs/
│   ├── base.yaml                 # 基础配置（所有默认超参数）
│   ├── debug_calvin.yaml         # 调试用小规模配置
│   └── ablation/                 # 消融实验配置
│       ├── no_gating.yaml        #   去除双向门控
│       ├── mse_energy.yaml       #   MSE 替代 InfoNCE 能量
│       └── mamba1_backbone.yaml  #   Mamba-1 替代 Mamba-3
├── src/
│   ├── config/                   # 配置管理与 dataclass schema
│   ├── data/                     # CALVIN 数据加载、摄动策略
│   ├── encoders/                 # 对比预训练视觉-本体编码器（ResNet18/ViT-Small）
│   ├── world_model/              # Mamba-3 CE-WM（能量头 + 复数域 SSM 核心层）
│   ├── steering/                 # EFE 偏转（朗之万动力学 + 双向门控）
│   ├── dual_stream/              # 非对称双流推理拓扑编排
│   ├── training/                 # 预训练管线（InfoNCE + NCE 损失）
│   ├── evaluation/               # 评估框架与指标
│   └── utils/                    # 检查点、日志、可视化
├── scripts/
│   ├── convert_calvin_mmap.py    # 数据格式转换（npz → mmap，加速训练 I/O）
│   ├── pretrain.py               # 两阶段预训练启动脚本
│   ├── run_paper_experiments.py  # 主实验（CE-AIS vs 4 baselines）
│   ├── ablation.py               # 消融实验
│   ├── evaluate.py               # 单次评估
│   ├── evaluate_ood.py           # OOD 鲁棒性评估
│   ├── evaluate_recovery.py      # 恢复能力评估
│   ├── evaluate_pareto.py        # 延迟-性能帕累托曲线
│   └── plot_results.py           # 论文图表生成
├── tests/                        # 单元测试 + 属性测试 (Hypothesis)
└── data/
    └── task_ABC_D/               # CALVIN ABC→D 数据集（~524GB）
        ├── training/             #   训练集（147 episodes, ~179 万帧）
        └── validation/           #   验证集
```

## 环境安装

要求 Python >= 3.10，CUDA >= 11.8，推荐使用 [uv](https://docs.astral.sh/uv/) 包管理器。

```bash
# 基础安装
uv sync

# 如需 OpenVLA 真实推理
uv sync --extra vla

# 如需量化支持（24GB 显存紧凑场景）
uv sync --extra vla --extra quant
```

## 数据集

本项目使用 [CALVIN](https://github.com/mees/calvin) 基准数据集，这是一个面向语言条件长周期机器人操作的评估平台。每个时间步包含 RGB 图像 (200x200)、深度图、机器人本体状态和相对动作。

### 下载

| 数据集 | 说明 | 大小 | 下载链接 |
|--------|------|------|----------|
| **task_ABC_D** | ABC→D 跨环境泛化（本项目主实验） | ~517 GB | http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip |
| **task_D_D** | D→D 单环境基线 | ~166 GB | http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip |
| **task_ABCD_D** | ABCD→D 全环境 | ~656 GB | http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip |
| **debug** | 调试用小数据集 | ~1.3 GB | http://calvin.cs.uni-freiburg.de/dataset/debug.zip |

```bash
# 推荐使用 aria2c 多线程下载（大文件更稳定）
aria2c -x 16 -s 16 -k 1M -c http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip

# 或用 wget
wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip

# 解压到 data/ 目录
unzip task_ABC_D.zip -d data/

# 调试时可先用 debug 数据集验证流程
wget http://calvin.cs.uni-freiburg.de/dataset/debug.zip
unzip debug.zip -d data/
```

下载后目录结构：

```
data/task_ABC_D/
├── training/              # 训练集（147 episodes, ~179 万帧）
│   ├── episode_0000000.npz
│   ├── episode_0000001.npz
│   ├── ...
│   └── ep_start_end_ids.npy
└── validation/            # 验证集
```

## 快速验证

```bash
# 运行全部测试（不需要 GPU 和数据集）
uv run python -m pytest tests/ -v

# 仅单元测试
uv run python -m pytest tests/unit/ -v

# 属性测试
uv run python -m pytest tests/property/ -v --hypothesis-seed=42
```

---

## 完整训练流程

整个实验分四步：**数据准备 → 预训练 → 跑实验 → 出图**。

### Step 0: 数据格式转换（强烈推荐）

CALVIN 数据集由 ~179 万个独立 `.npz` 压缩文件组成，原生加载时每个 batch 要打开数百个文件并逐个解压 zip，导致 GPU 严重空转。

转换为**内存映射（mmap）格式**后，读取变为数组索引操作，训练 I/O 吞吐提升一个数量级。

```bash
# 转换训练集（多进程并行，约 15-20 分钟）
# --workers: 并行进程数（默认 CPU 核数的一半，可自行指定）
uv run python scripts/convert_calvin_mmap.py \
    --src data/task_ABC_D/training \
    --dst /tmp/calvin_mmap/training \
    --workers 24

# 如需转换验证集
uv run python scripts/convert_calvin_mmap.py \
    --src data/task_ABC_D/validation \
    --dst /tmp/calvin_mmap/validation \
    --workers 24
```

转换后 `configs/base.yaml` 中的 `data.mmap_dir` 已指向 `/tmp/calvin_mmap`，训练时会自动检测并使用 mmap 后端。如果 mmap 目录不存在，会自动回退到原生 `.npz` 加载。

> **注意**:
> - `/tmp` 目录在系统重启后会被清空，重启后需要重新转换
> - 如果有其他持久存储空间，改 `--dst` 路径并同步修改 `base.yaml` 中的 `mmap_dir`
> - 需要目标磁盘有 ~360GB 可用空间

### Step 1: 两阶段预训练

```bash
# 完整预训练（Stage 1: Encoder 100 epochs + Stage 2: CE-WM 200 epochs）
uv run python scripts/pretrain.py --config configs/base.yaml
```

多卡并行训练（使用 HuggingFace Accelerate）：

```bash
# 2 卡并行（自动数据并行 DDP + 混合精度）
c

# 指定 GPU 设备
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes=2 \
    scripts/pretrain.py --config configs/base.yaml
```

> **注意**：batch_size 配置为**每卡**批大小。2 卡并行时实际有效 batch_size = 配置值 × 2。
> 单卡运行 `uv run python scripts/pretrain.py` 仍然完全兼容，无需任何改动。

分阶段执行：

```bash
# 仅预训练 Encoder（Stage 1）
uv run python scripts/pretrain.py --config configs/base.yaml --stage encoder

# 仅预训练 CE-WM（Stage 2，需要 Encoder 权重，自动从 checkpoints/ 加载）
uv run python scripts/pretrain.py --config configs/base.yaml --stage cewm

# 指定 Encoder 权重路径
uv run python scripts/pretrain.py --config configs/base.yaml --stage cewm \
    --encoder-ckpt checkpoints/encoder_epoch100.pt

# 从断点恢复训练
uv run python scripts/pretrain.py --config configs/base.yaml --resume

# 覆盖配置参数（例如调小 batch_size 避免 OOM）
uv run python scripts/pretrain.py --config configs/base.yaml \
    --override training.batch_size=128 training.encoder_epochs=50
```

训练产物：
- `checkpoints/encoder_epoch*.pt` — Encoder 权重
- `checkpoints/cewm_epoch*.pt` — CE-WM 权重
- `logs/` — TensorBoard 日志

```bash
# 查看训练曲线
tensorboard --logdir logs/
```

### Step 2: 主实验

```bash
# CE-AIS vs 4 baselines（CALVIN ABC→D 协议）
PYOPENGL_PLATFORM=egl uv run python scripts/run_paper_experiments.py \
    --vla-type openvla
```

### Step 3: 消融实验

```bash
# 跑全部 3 组消融（no_gating, mse_energy, mamba1_backbone）
PYOPENGL_PLATFORM=egl uv run python scripts/ablation.py \
    --config configs/base.yaml

# 跑指定消融
PYOPENGL_PLATFORM=egl uv run python scripts/ablation.py \
    --config configs/base.yaml --variants no_gating mse_energy

# 列出可用消融配置
uv run python scripts/ablation.py --config configs/base.yaml --list
```

### Step 4: 生成论文图表

```bash
uv run python scripts/plot_results.py

# 输出:
#   figures/main_table.tex       — 主实验对比表（表 3）
#   figures/ood_table.tex        — OOD 实验对比表（表 4）
#   figures/recovery_curve.pdf   — U 型反弹恢复曲线（图 5）
```

---

## 其他评估脚本

```bash
# OOD 鲁棒性评估（物理/视觉摄动）
PYOPENGL_PLATFORM=egl uv run python scripts/evaluate_ood.py

# 恢复能力评估
PYOPENGL_PLATFORM=egl uv run python scripts/evaluate_recovery.py

# 延迟-性能帕累托曲线
uv run python scripts/evaluate_pareto.py
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.batch_size` | 256 | 训练批大小（RTX 3090 24GB 推荐 256） |
| `training.encoder_epochs` | 100 | Encoder 预训练轮数 |
| `training.ce_wm_epochs` | 200 | CE-WM 预训练轮数 |
| `training.learning_rate` | 1e-4 | AdamW 学习率 |
| `training.amp` | true | 混合精度训练 |
| `ce_wm.n_layers` | 32 | Mamba-3 层数（32 层 ~122M params） |
| `ce_wm.d_model` | 640 | 模型隐藏维度 |
| `steering.n_steps` | 5 | 朗之万迭代步数 |
| `steering.grad_mode` | finite_diff | 梯度模式：finite_diff（快）或 autograd（精确） |
| `bilateral_gating.mc_samples` | 5 | MC-Dropout 采样次数 |
| `data.num_workers` | 24 | DataLoader 工作进程数 |
| `data.mmap_dir` | /tmp/calvin_mmap | mmap 加速数据目录 |

## 性能参考（RTX 3090 24GB）

| 指标 | 数值 |
|------|------|
| CE-WM 单次前向 (T=1) | ~1.5 ms |
| 完整偏转流程 (n_steps=5) | ~21 ms |
| 推理帧率 | ~48 FPS |
| CE-WM 参数量 | 122.3M |

## 架构概览

```
观测 ──→ [VLA 策略 (冻结)] ──→ 候选动作 a_init
  │                                 │
  └──→ [Encoder (冻结)] ──→ z_t     │
                             │      ↓
                      [CE-WM: Mamba-3 SSM (冻结)]
                             │
                      能量 E(z,a) + 不确定性 u
                             │
                      [双向门控] → λ(u)
                             │
                      [退火朗之万偏转] → a*（校正动作）
```

- **Encoder**: ResNet18 视觉骨干 + 本体感觉融合，InfoNCE 对比预训练
- **CE-WM**: 32 层 Mamba-3（复数域 SSM + MIMO 分组），NCE 能量景观训练
- **偏转**: 退火朗之万动力学，有限差分梯度估计（避免 autograd 开销）
- **门控**: 双向认知不确定性门控，MC-Dropout 估计

## 磁盘空间管理

如果磁盘空间紧张，以下缓存可以安全清理：

```bash
pip cache purge        # 清理 pip 缓存（~21GB）
uv cache clean         # 清理 uv 缓存（~50GB）
```

## 许可证

MIT
