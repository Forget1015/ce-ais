# CE-AIS: Causal-Energy Active Inference Steering

无梯度测试时自适应框架，用于具身智能 VLA 策略的在线校正。

核心思路：冻结 VLA 参数，通过因果能量世界模型 (CE-WM) 在推理时评估候选动作的能量。当前版本将 CE-AIS 从 always-on 动作偏转器升级为**安全/可弃权的动作审查与恢复控制器**：只有当 trust-region 内的候选动作带来更低能量且不确定性可接受时才执行干预，否则回退到冻结 VLA 原始动作。

## Safe CE-AIS 更新要点

- **冻结不变**：VLA、Encoder、CE-WM 在测试时全部 `requires_grad=False`，只搜索动作张量，不更新参数。
- **Trust region**：`steering.action_delta_max` 限制 CE-AIS 相对 VLA 动作的每维最大偏移，优先保证 clean non-degradation。
- **Accept/reject**：`energy_after <= energy_before - accept_energy_margin` 才接受 steered action，否则直接执行 VLA prior。
- **不确定性弃权**：`bilateral_gating.hard_uncertainty_threshold` 可在 CE-WM 不确定性过高时强制 abstain。
- **诊断输出**：主实验和 OOD JSON 中的 `ce_ais_diagnostics` 会记录 accepted/rejected/abstained/fallback 比率、平均能量、不确定性、门控强度和动作偏移。
- **CE-WM 校准训练**：NCE loss 现在加入能量尺度正则与目标 margin 约束，避免 margin 爆炸后坍缩到 `ln(1+K)` 随机分类状态。

推荐先使用早期且 margin 稳定的 CE-WM checkpoint（例如已有实验中的 `checkpoints_sub/cewm_epoch0015.pt` 或相邻 checkpoint）做评估；若训练日志出现极大 margin 或 loss 接近 `ln(6)=1.7918`，不要直接把该 checkpoint 当成可靠 steering 场。

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
# --workers: 并行进程数（默认 CPU 核数的一半，可自行指定） nproc lscpu
uv run python scripts/convert_calvin_mmap.py \
    --src data/task_ABC_D/training \
    --dst data/calvin_mmap/training \
    --workers 56

# 如需转换验证集
uv run python scripts/convert_calvin_mmap.py \
    --src data/task_ABC_D/validation \
    --dst data/calvin_mmap/validation \
    --workers 56
```

转换后 `configs/base.yaml` 中的 `data.mmap_dir` 已指向 `data/calvin_mmap`，训练时会自动检测并使用 mmap 后端。如果 mmap 目录不存在，会自动回退到原生 `.npz` 加载。

> **注意**:
> - 默认转换到 `data/calvin_mmap`，这是持久路径，不会因为系统重启被清空
> - 如果使用其他存储路径，改 `--dst` 路径并同步修改 `base.yaml` 中的 `mmap_dir`
> - 需要目标磁盘有 ~360GB 可用空间

### Step 1: 两阶段预训练

```bash
# 完整预训练（Stage 1: Encoder 30 epochs + Stage 2: CE-WM 200 epochs）
uv run python scripts/pretrain.py --config configs/base.yaml
```

多卡并行训练（使用 HuggingFace Accelerate）：

```bash
# 2 卡并行（自动数据并行 DDP + 混合精度）
uv run accelerate launch --num_processes=8 scripts/pretrain.py \
    --config configs/base.yaml

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
    --encoder-ckpt checkpoints/encoder_epoch0030.pt

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
# 不指定 checkpoint 时，会自动加载 checkpoints/ 下最新的 encoder_epoch*.pt 和 cewm_epoch*.pt
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
    --vla-type openvla

# 推荐：显式指定 GPU、Encoder checkpoint 和 CE-WM checkpoint
# 默认使用官方 CALVIN 可达任务序列：固定 initial_state + 5 个可连续完成的 subtask + 自然语言指令
# --device cuda:N 直接选择物理 GPU N；不要再同时设置 CUDA_VISIBLE_DEVICES
# 脚本会自动把 EGL_VISIBLE_DEVICES 设为同一个 GPU 序号，避免 PyTorch 和 EGL 分别占用不同卡
# HF_HOME 指向本机已有 OpenVLA cache，离线模式可避免访问 HuggingFace / hf-mirror
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
    --vla-type openvla \
    --device cuda:6 \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints/cewm_epoch0015.pt

# FLOWER VLA 主实验：先用 --n-chains 20 做 smoke；确认无退化后再用 --n-chains 200 出正式表
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
uv run python scripts/run_paper_experiments.py \
    --data-dir data/task_ABC_D \
    --vla-type flower \
    --flower-checkpoint-dir data/flower_calvin_abc \
    --flower-code-path external/flower_vla_calvin \
    --methods frozen_flower ce_ais \
    --sequence-source official \
    --n-chains 200 \
    --chain-length 5 \
    --max-steps 360 \
    --device cuda:6 \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints_sub/cewm_epoch0015.pt

# 如果当前机器缺少 pybullet 的 eglRendererPlugin，加 --no-egl 使用 DIRECT/TinyRenderer
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
    --vla-type openvla \
    --device cuda:4 \
    --no-egl \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints/cewm_epoch0015.pt
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONUNBUFFERED=1 \
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
uv run accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  scripts/pretrain.py \
  --config configs/base.yaml \
  --stage cewm \
  --resume
  --encoder-ckpt checkpoints/encoder_epoch0044.pt \
  --checkpoint-dir checkpoints_calibrated_cewm \
  --log-dir logs/calibrated_cewm_multigpu \
  --override \
  training.ce_wm_epochs=100 \
  training.cewm_batch_size=256 \
  training.learning_rate=1.0e-4 \
  training.energy_reg_weight=1.0e-4 \
  training.target_margin=5.0 \
  training.min_margin=1.0 \
  training.margin_upper_weight=1.0e-2 \
  training.margin_lower_weight=1.0 \
  training.monitor_action_grad_norm=true \
  training.checkpoint_interval=1 \
  data.num_workers=8 \
2>&1 | tee logs/calibrated_cewm_multigpu_10epoch.log
```

本机 OpenVLA cache 位于 `/data0/yejinxuan/hf_cache/hub/models--openvla--openvla-7b`，包含 `processor_config.json`、`model.safetensors.index.json` 和 3 个 `model-*.safetensors` 分片；设置 `HF_HOME=/data0/yejinxuan/hf_cache` 后，`openvla/openvla-7b` 会从该 cache 读取。若需要重新下载或更新模型，去掉 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 并确保代理/网络可用。

主实验默认 `--sequence-source official`，会按 CALVIN 官方长程评估方式生成可达任务链，并把 task key 映射为自然语言指令。旧的随机任务链可用 `--sequence-source random` 复现，但它会随机组合任务和初始状态，通常不适合报告成功率。

`eglRendererPlugin` 是 PyBullet/CALVIN 在无显示器服务器上做 GPU headless 渲染的插件，负责更快地渲染相机 RGB/depth 观测。缺失它不改变已训练的 Encoder、CE-WM、VLA 权重，也不改变任务定义；使用 `--no-egl` 会退回 PyBullet DIRECT/TinyRenderer，主要影响是渲染速度可能更慢，视觉观测可能和 EGL 渲染存在轻微数值差异。为了结果可比，同一组主实验和 baseline 应统一使用同一种渲染模式。

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

下面三个脚本已统一走 `VLAAdapter` 接口。新增模型时，只要在 `build_vla_adapter(config)` 中注册 adapter，并把模型类型加入 `SUPPORTED_VLA_TYPES`，这些评估脚本就能通过 `--vla-type` 切换。

```bash
# OOD severity sweep：physics / visual / camera × mild / medium / severe
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
PYTHONUNBUFFERED=1 \
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
uv run python -u scripts/evaluate_ood.py \
    --data-dir data/task_ABC_D \
    --vla-type flower \
    --flower-checkpoint-dir data/flower_calvin_abc \
    --flower-code-path external/flower_vla_calvin \
    --methods frozen_flower ce_ais \
    --ood-types physics visual camera \
    --severity-sweep \
    --severities mild medium severe \
    --n-episodes 100 \
    --chain-length 5 \
    --max-steps 360 \
    --progress-interval 5 \
    --device cuda:0 \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints_sub/cewm_epoch0015.pt 2>&1 | tee /data0/yejinxuan/ce-ais/logs/ood_severity_$(date +%Y%m%d_%H%M%S).log

# 恢复能力评估：在 inject-step 注入 OOD 后观察恢复曲线
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
uv run python scripts/evaluate_recovery.py \
    --data-dir data/task_ABC_D \
    --vla-type flower \
    --flower-checkpoint-dir data/flower_calvin_abc \
    --flower-code-path external/flower_vla_calvin \
    --methods frozen_flower ce_ais \
    --ood-type physics \
    --n-episodes 100 \
    --max-steps 120 \
    --inject-step 60 \
    --device cuda:6 \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints_sub/cewm_epoch0015.pt

# 延迟-性能帕累托曲线：扫描 CE-AIS n_steps
HF_HOME=/data0/yejinxuan/hf_cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYOPENGL_PLATFORM=egl \
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
uv run python scripts/evaluate_pareto.py \
    --data-dir data/task_ABC_D \
    --vla-type flower \
    --flower-checkpoint-dir data/flower_calvin_abc \
    --flower-code-path external/flower_vla_calvin \
    --baseline-methods frozen_flower \
    --ce-ais-n-steps 1 3 5 \
    --n-chains 50 \
    --chain-length 5 \
    --max-steps 360 \
    --device cuda:6 \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints_sub/cewm_epoch0015.pt
```

常用 adapter 参数：

| 参数 | 用于 | 说明 |
|------|------|------|
| `--vla-type` | 全部 | 选择 `proxy` / `openvla` / `flower` / `calvin` 或后续新增 adapter |
| `--device` | 全部 | 评估设备；`cuda:N` 直接选择物理 GPU N |
| `--encoder-ckpt` | CE-AIS | 指定 Encoder checkpoint |
| `--cewm-ckpt` | CE-AIS | 指定 CE-WM checkpoint |
| `--flower-checkpoint-dir` | FLOWER | FLOWER checkpoint 目录，包含 `config.yaml` 和 `model.safetensors` |
| `--flower-code-path` | FLOWER | FLOWER 官方代码 checkout 路径 |
| `--calvin-policy-ckpt` | CALVIN policy | CALVIN 原生 policy checkpoint |
| `--calvin-train-folder` | CALVIN policy | 含 `.hydra/config.yaml` 的训练目录 |
| `--calvin-dataset-path` | CALVIN policy | CALVIN 数据集路径，默认跟随 `--data-dir` |
| `--no-egl` | 全部 | 关闭 EGL，回退 DIRECT/TinyRenderer |

`frozen_vla`、`frozen_openvla` 和 `frozen_flower` 都是冻结当前 `--vla-type` adapter 的 baseline 名称；推荐新实验用语义更清楚的 `frozen_vla` 或模型专用别名。

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.batch_size` | 256 | 训练批大小（RTX 3090 24GB 推荐 256） |
| `training.encoder_epochs` | 30 | Encoder 预训练轮数 |
| `training.ce_wm_epochs` | 200 | CE-WM 预训练轮数 |
| `training.learning_rate` | 1e-4 | AdamW 学习率 |
| `training.amp` | true | 混合精度训练 |
| `ce_wm.n_layers` | 32 | Mamba-3 层数（32 层 ~122M params） |
| `ce_wm.d_model` | 640 | 模型隐藏维度 |
| `steering.n_steps` | 1 | 安全默认朗之万迭代步数 |
| `steering.grad_mode` | finite_diff | 梯度模式：finite_diff（快）或 autograd（精确） |
| `steering.action_delta_max` | 0.05 | 相对 VLA prior 的每维最大动作偏移 |
| `steering.enable_accept_reject` | true | 能量未改善时回退 VLA 原动作 |
| `steering.accept_energy_margin` | 0.0 | 接受干预所需的最小能量下降 |
| `bilateral_gating.lambda_max` | 0.2 | 安全默认最大引导强度 |
| `bilateral_gating.hard_uncertainty_threshold` | null | 高不确定性强制 abstain 阈值 |
| `bilateral_gating.mc_samples` | 5 | MC-Dropout 采样次数 |
| `training.energy_reg_weight` | 1e-4 | CE-WM 能量尺度正则 |
| `training.target_margin` | 5.0 | CE-WM 目标能量 margin |
| `training.margin_upper_weight` | 1e-2 | margin 过大惩罚权重 |
| `training.margin_lower_weight` | 1.0 | margin 不足惩罚权重 |
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
                      [Trust-region EFE] → a*
                             │
                      [Accept/Reject] → 执行 a* 或回退 a_init
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
