# LIBERO 实验启动详细步骤

本文档列出 CE-AIS 接入 LIBERO benchmark 的完整执行步骤，包括每一步需要下载什么、从哪里下载、如何验证。

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

---

## Phase 1: 环境与数据准备

### Step 1: LIBERO 环境安装与验证

LIBERO 已经作为 submodule 存在于 `external/flower_vla_calvin/LIBERO/`。

**1.1 确认 submodule 状态**

```bash
cd /data0/yejinxuan/ce-ais
git submodule update --init --recursive
ls external/flower_vla_calvin/LIBERO/libero/
```

验证：应看到 `libero/` 目录下有 `benchmark/`, `envs/`, `lifelong/` 等子目录。

**1.2 安装 LIBERO**

```bash
cd /data0/yejinxuan/ce-ais/external/flower_vla_calvin/LIBERO
pip install -e .
```

LIBERO 依赖：
- robosuite==1.4.0（内含 mujoco 渲染）
- mujoco（需要 MuJoCo 2.1+ 或 DeepMind mujoco>=2.3.0）
- numpy, torch, hydra-core, gym==0.25.2

**1.3 验证 LIBERO 能正常 import**

```python
python -c "
from libero.libero import benchmark, get_libero_path
print('LIBERO benchmark suites:', list(benchmark.get_benchmark_dict().keys()))
print('BDDL path:', get_libero_path('bddl_files'))
print('Init states path:', get_libero_path('init_states'))
"
```

验证：应输出包含 `libero_spatial`, `libero_object`, `libero_goal`, `LIBERO_10`, `LIBERO_90` 的列表。

**1.4 验证渲染**

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
cd /data0/yejinxuan/ce-ais/external/flower_vla_calvin/LIBERO
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```
这会下载所有 suite 到 `~/.libero/datasets/` 或 LIBERO 默认路径。

方式 B：HuggingFace 手动下载
- 地址：https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
```bash
# 用 huggingface-cli
pip install huggingface_hub
huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset --local-dir /data0/yejinxuan/ce-ais/data/libero_datasets
```

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

**7.1 需要实现的脚本**

创建 `src/data/libero_dataset.py`，参考现有 `src/data/calvin_dataset.py`。

核心逻辑：
```python
class LIBERODataset(Dataset):
    """从 LIBERO HDF5 demo 中加载训练数据"""

    def __init__(self, data_dir, suite_name, mode="ce_wm", window_size=16):
        # 加载所有 demo 轨迹
        # mode="encoder": 返回 (anchor_obs, positive_obs) 用于对比学习
        # mode="ce_wm": 返回 (obs_seq, action_seq) 用于能量模型训练
        ...

    def _load_demos(self):
        """从 HDF5 文件加载所有 demo"""
        for hdf5_file in self.hdf5_files:
            f = h5py.File(hdf5_file, 'r')
            for demo_key in f['data'].keys():
                demo = f['data'][demo_key]
                actions = demo['actions'][:]  # (T, 7)
                obs = {
                    'agentview_image': demo['obs/agentview_image'][:],
                    'robot0_joint_pos': demo['obs/robot0_joint_pos'][:],
                    'robot0_gripper_qpos': demo['obs/robot0_gripper_qpos'][:],
                }
                self.trajectories.append((obs, actions))
```

**7.2 数据量估算**

- 4 个主 suite × 10 task × 50 demo × ~200 steps/demo ≈ 400k frames
- 足够训练 encoder 和 CE-WM

**7.3 验证**

```python
from src.data.libero_dataset import LIBERODataset
ds = LIBERODataset("/data0/yejinxuan/ce-ais/data/libero_datasets", "libero_spatial", mode="ce_wm")
print(f"Total samples: {len(ds)}")
sample = ds[0]
print(f"obs_seq shape: {sample['obs_seq'].shape}")  # (T, obs_dim)
print(f"action_seq shape: {sample['action_seq'].shape}")  # (T, 7)
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

建议策略：
- **论文主实验**：per-suite CE-WM（在每个 suite 的 demo 上分别训练）
- **泛化实验**：cross-suite CE-WM（在 libero_90 上训练，在 libero_10 上评估）

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

- [ ] LIBERO 环境能 import 并渲染（Step 1 完成）
- [ ] LIBERO 数据集下载完毕且能读取（Step 2 完成）
- [ ] FLOWER LIBERO checkpoint 下载并加载成功（Step 3 完成）
- [ ] FLOWER on LIBERO baseline 跑通，成功率合理（Step 4 完成）
- [ ] `libero_integration.py` 编写完成并测试（Step 6 完成）
- [ ] CE-WM 在 LIBERO demo 上训练完成（Step 8 完成）
- [ ] CE-AIS + FLOWER LIBERO 联合评估有结果（Step 9 完成）
- [ ] OpenVLA-OFT baseline 跑通（Step 10 完成）
- [ ] 至少两个 baseline + CE-AIS 的 LIBERO 主表格填满（Phase 4 完成）
