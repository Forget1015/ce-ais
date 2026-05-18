# CALVIN 多 VLA + CE-AIS 实验技术路线

## 1. 目标

当前 `openvla/openvla-7b` zero-shot 跑 CALVIN 为 0%，本地 `D_D_static_rgb_baseline` 在 `task_ABC_D` 官方链路上也未跑出成功率。因此后续论文实验不应继续依赖这两个模型作为主结果。

新的实验目标是：

> 选择多个已经在 CALVIN 上训练或微调过的 VLA / action model，先验证其 frozen baseline 有非零成功率，再比较 `Frozen` 与 `Frozen + CE-AIS` 在 clean 和 OOD 场景下的表现差异。

论文叙事应从“CE-AIS 拯救 zero-shot OpenVLA”调整为：

> CE-AIS 是一个不修改基座模型参数的 test-time action-space steering 模块，可以插到多个 frozen action-compatible VLA / action model 后面，在不确定性、分布偏移和扰动恢复场景中提升鲁棒性。

## 2. 总体实验流程

本项目后续实验要按“**同一 CALVIN 评测协议、同一 VLAAdapter 接口、同一 CE-AIS 后处理模块**”组织。不同 VLA / action model 只允许在 adapter 内部切换，外层评测脚本、OOD/recovery 设计和 CE-AIS steering 逻辑保持一致。

统一目标接口：

```python
class VLAAdapter:
    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        return actions  # Tensor[B, T, 7], CALVIN rel_actions
```

本地已下载 checkpoint 的 split 判断：

| Model | 本地路径 | 当前可确认 split | 对论文实验是否推荐 | 说明 |
|---|---|---|---|---|
| RoboFlamingo | `data/RoboFlamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_4.pth` | **本地文件名/权重扫描未直接标明 ABC 或 ABCD** | 推荐，但必须先官方 eval 验证 | 文件是 CALVIN 风格 RoboFlamingo checkpoint，但不能仅凭本地文件确认是 ABC→D 还是 ABCD→D。 |
| FLOWER VLA | `data/flower_calvin_abc/model.safetensors` | **ABC 训练，对应 `task_ABC_D` / ABC→D** | 强烈推荐 | README 与 config 均写明 `flower_calvin_abc`、`benchmark_name: calvin_abc`、`root_data_dir: task_ABC_D`，输出 `(B,T,7)` delta EEF action，最适合先接。 |
| RoboVLMs | `data/robovlms/checkpoints/kosmos_ph_calvin_abc.pt` | **ABC 训练，对应 `task_ABC_D` / ABC→D** | 推荐作为第三个泛化模型 | 本地只有 `kosmos_ph_calvin_abc.pt`；虽然有 `kosmos_ph_calvin_abcd.json`，但没有对应 `kosmos_ph_calvin_abcd.pt` 权重。 |

结论：你现在下载的 FLOWER VLA 和 RoboVLMs 可以确认是 **ABC→D 主实验方向**；RoboFlamingo 暂时只能确认是 CALVIN/RoboFlamingo 风格 checkpoint，不能从本地元数据确认 split，必须用官方 evaluation 标定后再作为主结果。

每个推荐模型都按同一“五步走”接入：

### Step 1：官方代码与权重验收

- **RoboFlamingo**：保留当前 `.pth`，补齐官方 RoboFlamingo 代码和运行配置，先确认该 checkpoint 对应的 CALVIN split。
- **FLOWER VLA**：使用 `flower_calvin_abc`，记录 `config.yaml` 中的 `task_ABC_D`、`act_seq_len=10`、`rel_actions`、双视角输入。
- **RoboVLMs**：使用 `kosmos_ph_calvin_abc.pt` + `configs/kosmos_ph_calvin_abc.json`，不要误用没有本地权重的 `kosmos_ph_calvin_abcd.json`。

输出记录：`model_name / checkpoint_path / split / action_horizon / action_format / official_eval_command`。

### Step 2：先跑官方 evaluation，建立 frozen 上限

必须先在各自官方仓库跑 CALVIN long-horizon evaluation：

- 如果官方 eval 在 `task_ABC_D` 上非零，才进入 adapter 阶段；
- 如果官方 eval 为 0，先排查 split、语言标注、相机输入、action unnormalization、gripper convention；
- 不允许直接把官方 eval 失败的模型接入 CE-AIS，否则论文结果无法解释。

输出记录：`L1/L2/L3/L4/L5 / AvgLen / latency / GPU memory`。

### Step 3：写统一 frozen adapter

每个模型只新增一个 adapter 类型，全部收敛到同一个接口：

| Model | Adapter 名称 | `--vla-type` 建议 | 输出约定 |
|---|---|---|---|
| RoboFlamingo | `RoboFlamingoAdapter` | `roboflamingo` | `Tensor[B,T,7]`，必要时从单步 action 扩成 chunk |
| FLOWER VLA | `FlowerVLAAdapter` | `flower` | 原生 `(B,T,7)`，优先保留 `act_seq_len=10` |
| RoboVLMs | `RoboVLMsAdapter` | `robovlms` | `inference_step(...)["action"]` 后 unnormalize 到 CALVIN rel_actions |

adapter 内部负责：图像 resize/normalize、static/gripper 双视角映射、instruction prompt、action unnormalization、hidden state/reset、gripper 符号转换。adapter 外部只看 `predict(observation, instruction) -> Tensor[B,T,7]`。

### Step 4：用本项目评测脚本复现 frozen baseline

把三个模型都接入 `build_vla_adapter(config)` 后，用同一命令模板切换：

```bash
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
  --data-dir data/task_ABC_D \
  --vla-type <roboflamingo|flower|robovlms> \
  --methods frozen_<model> \
  --sequence-source official \
  --n-chains 100 \
  --chain-length 5 \
  --max-steps 360 \
  --device cuda:0
```

通过标准：本项目 frozen 结果应接近官方 eval。若差距很大，优先修 adapter，不要先调 CE-AIS。

### Step 5：同一 CE-AIS 模块做 clean / OOD / recovery 对比

frozen baseline 对齐后，再统一接 CE-AIS：

```bash
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
  --data-dir data/task_ABC_D \
  --vla-type <roboflamingo|flower|robovlms> \
  --methods frozen_<model> ce_ais \
  --sequence-source official \
  --n-chains 100 \
  --chain-length 5 \
  --max-steps 360 \
  --device cuda:0
```

论文主对比固定为：

```text
Frozen(model) vs Frozen(model) + CE-AIS
```

核心指标固定为：clean 不明显掉点、visual/action OOD 提升、recovery 提升、latency 可接受、gating 触发率可解释。这样 RoboFlamingo / FLOWER / RoboVLMs 都是同一个实验接口，后续只改 `--vla-type` 和 checkpoint 配置即可随意切换。

## 3. 推荐模型优先级

### 3.1 第一优先级：RoboFlamingo

用途：经典 CALVIN VLA 强基线。

推荐原因：

- CALVIN 领域经典模型；
- 有公开 CALVIN checkpoint；
- 审稿人熟悉，作为主 baseline 有说服力；
- 适合验证 CE-AIS 能否改善老牌大模型 policy。

代码与权重：

- GitHub: `https://github.com/RoboFlamingo/RoboFlamingo`
- Hugging Face: `https://huggingface.co/robovlms/RoboFlamingo`

建议优先下载 ABC→D 或 ABCD→D 相关 checkpoint。权重较大，下载前确认磁盘空间。

推荐本地目录：

```bash
/data0/yejinxuan/ce-ais/external/roboflamingo
/data0/yejinxuan/ce-ais/data/vla_checkpoints/roboflamingo
```

### 3.2 第二优先级：FLOWER VLA

用途：新型 flow-matching / diffusion action model 强基线。

推荐原因：

- 技术路线新，适合顶会论文对比；
- 有 CALVIN 专用权重；
- 如果 CE-AIS 能改善它，论文说服力强；
- 适合验证 CE-AIS 对非自回归 action policy 是否有效。

代码与权重：

- GitHub: `https://github.com/intuitive-robots/flower_vla_calvin`
- Hugging Face collection: `https://huggingface.co/collections/mbreuss/flower-vla`
- CALVIN ABC: `https://huggingface.co/mbreuss/flower_calvin_abc`
- CALVIN ABCD: `https://huggingface.co/mbreuss/flower_calvin_abcd`
- CALVIN D: `https://huggingface.co/mbreuss/flower_calvin_d`

推荐本地目录：

```bash
/data0/yejinxuan/ce-ais/external/flower_vla_calvin
/data0/yejinxuan/ce-ais/data/vla_checkpoints/flower
```

### 3.3 第三优先级：RoboVLMs

用途：多 VLM/VLA 集成框架 baseline。

推荐原因：

- 本身就是比较不同 VLM/VLA 的框架；
- 支持 CALVIN；
- 有公开 CALVIN checkpoints；
- 很适合论文中做“多模型泛化”实验。

代码与权重：

- GitHub: `https://github.com/Robot-VLAs/RoboVLMs`
- Hugging Face: `https://huggingface.co/robovlms/RoboVLMs`

可关注 checkpoint：

- `kosmos_ph_calvin_abcd`
- `kosmos_ph_calvin_abc`

推荐本地目录：

```bash
/data0/yejinxuan/ce-ais/external/RoboVLMs
/data0/yejinxuan/ce-ais/data/vla_checkpoints/robovlms
```

### 3.4 第二阶段候选：UniVLA / UD-VLA

用途：现代 7B/8B 级 VLA 强基线。

UniVLA：

- GitHub: `https://github.com/OpenDriveLab/UniVLA`
- CALVIN checkpoint: `https://huggingface.co/qwbu/univla-7b-224-sft-calvin`

UD-VLA：

- GitHub: `https://github.com/OpenHelix-Team/UD-VLA`
- CALVIN checkpoint: `https://huggingface.co/chenpyyy/UD-VLA_CALVIN_ABCD_D`

推荐原因：

- 模型较新；
- 更贴近当前 VLA 研究趋势；
- 适合作为强现代 baseline。

风险：

- 模型大，显存和依赖压力更高；
- 接入成本可能明显高于 RoboFlamingo / FLOWER；
- 建议等前两个模型跑通后再做。

### 3.5 扩展候选：FALCON-VLA

用途：空间 token / 3D prior VLA baseline。

代码与权重：

- GitHub: `https://github.com/FALCON-VLA/FALCON`
- Hugging Face: `https://huggingface.co/FALCON-VLA/FALCON-series`

推荐原因：

- 代表空间理解增强的 VLA；
- 如果 CE-AIS 与 3D prior 模型也能互补，论文价值更高。

风险：

- 可能需要 depth、point cloud 或特殊输入；
- 适配成本最高；
- 不建议第一阶段接入。

## 4. 下载与环境准备

### 4.1 建议目录结构

在项目内建立统一外部模型目录：

```bash
/data0/yejinxuan/ce-ais/external/
/data0/yejinxuan/ce-ais/data/vla_checkpoints/
```

建议不要把外部模型代码混进 `src/`，避免污染 CE-AIS 主代码。

### 4.2 下载顺序

建议顺序：

```text
RoboFlamingo → FLOWER VLA → RoboVLMs → UniVLA/UD-VLA → FALCON
```

先不要一次性下载全部模型。每个模型都先完成“官方 eval 成功 → adapter 成功 → CE-AIS 成功”闭环，再接下一个。

### 4.3 Hugging Face 下载建议

如果使用 Hugging Face，建议统一缓存：

```bash
export HF_HOME=/data0/yejinxuan/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
```

如果本地已有权重，运行时可加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

但第一次下载不要设置 offline。

## 5. 每个模型的验证步骤

### Step A：官方 evaluation 验证

目标：确认模型官方代码 + 官方 checkpoint 能在 CALVIN 跑出非零成功率。

记录内容：

```text
model_name
checkpoint_path
CALVIN split: ABC→D / ABCD→D / D→D
n_chains
chain_length
max_steps
L1 / L2 / L3 / L4 / L5
average sequence length
latency
```

通过标准：

```text
L1 > 0，最好 L3/L5 也非零。
```

如果 L1 = 0：

- 先不要接 CE-AIS；
- 检查 checkpoint 是否对应当前 split；
- 检查是否用了正确 CALVIN dataset；
- 检查官方 README 的 eval 命令是否完整；
- 必要时换同仓库另一个 checkpoint。

### Step B：写 frozen adapter

每个模型都要封装成统一接口：

```python
class SomeVLAAdapter(VLAAdapter):
    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        ...
        return action  # [B, T, 7]
```

接口要求：

- 输入使用当前 CALVIN observation；
- 输出必须是 CALVIN `rel_actions`；
- shape 必须是 `[B, T, 7]`；
- gripper 最后一维后续会在 env step 中二值化为 `-1 / 1`；
- 不允许 silent fallback 到 proxy / OpenVLA；
- 若缺 checkpoint，要明确报错。

验证方式：

```bash
uv run python -m py_compile src/dual_stream/vla_adapter.py
```

再跑短 smoke：

```bash
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
  --data-dir data/task_ABC_D \
  --vla-type <new_model_type> \
  --methods <new_frozen_baseline> \
  --sequence-source official \
  --n-chains 1 \
  --chain-length 1 \
  --max-steps 20 \
  --progress-steps 5 \
  --device cuda:0
```

短 smoke 只验证不崩溃，不用于判断成功率。

### Step C：frozen adapter 对齐官方结果

用我们的评测脚本跑：

```bash
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
  --data-dir data/task_ABC_D \
  --vla-type <new_model_type> \
  --methods <new_frozen_baseline> \
  --sequence-source official \
  --n-chains 100 \
  --chain-length 5 \
  --max-steps 360 \
  --progress-steps 25 \
  --device cuda:0
```

通过标准：

```text
我们的 frozen 结果应接近官方 evaluation。
```

如果官方有成功率，我们这里 0：

- adapter observation transform 错；
- action unnormalization 错；
- instruction string 和官方不一致；
- reset / hidden state 处理不一致；
- action chunk 取法不一致；
- gripper convention 错。

此时不要继续跑 CE-AIS，先修 adapter。

### Step D：接 CE-AIS

确认 frozen adapter 正常后，跑：

```bash
PYTHONPATH=/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH \
PYOPENGL_PLATFORM=egl \
uv run python scripts/run_paper_experiments.py \
  --data-dir data/task_ABC_D \
  --vla-type <new_model_type> \
  --methods <new_frozen_baseline> ce_ais \
  --sequence-source official \
  --n-chains 100 \
  --chain-length 5 \
  --max-steps 360 \
  --progress-steps 25 \
  --device cuda:0
```

观察：

- CE-AIS 是否崩溃；
- CE-AIS latency 是否可接受；
- clean 成功率是否不下降太多；
- OOD 或 recovery 是否提升。

## 6. OOD / recovery 实验设计

CE-AIS 不一定能在 clean SOTA 上大幅提升。更合理的实验重点是：

### 6.1 Clean setting

目的：证明 CE-AIS 不破坏原模型。

指标：

```text
Frozen vs +CE-AIS 的 L1/L3/L5 差距不能明显变差。
```

### 6.2 Visual OOD

可做扰动：

- RGB noise；
- brightness / contrast shift；
- color jitter；
- random occlusion；
- background distractor。

目的：证明 CE-AIS 在视觉扰动下更稳。

### 6.3 Physics / action perturbation

可做扰动：

- action noise；
- initial state small perturbation；
- object pose perturbation；
- gripper delay / action repeat variation。

目的：证明 CE-AIS 能做 test-time correction。

### 6.4 Recovery setting

设计方式：

- 前若干步注入轻微错误动作；
- 或从偏离 expert trajectory 的状态开始；
- 比较 frozen 和 CE-AIS 是否能恢复任务。

这是最适合 CE-AIS 的实验之一。

## 7. 主论文表格建议

### 7.1 Clean 主表

| Model | Frozen L1 | Frozen L3 | Frozen L5 | +CE-AIS L1 | +CE-AIS L3 | +CE-AIS L5 | ΔAvg |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoboFlamingo | | | | | | | |
| FLOWER VLA | | | | | | | |
| RoboVLMs | | | | | | | |
| UniVLA / UD-VLA | | | | | | | |

### 7.2 OOD 主表

| Model | Setting | Frozen AvgLen | +CE-AIS AvgLen | Δ | Latency |
|---|---|---:|---:|---:|---:|
| RoboFlamingo | visual OOD | | | | |
| RoboFlamingo | action noise | | | | |
| FLOWER VLA | visual OOD | | | | |
| FLOWER VLA | action noise | | | | |

### 7.3 Ablation 表

| Variant | L1 | L3 | L5 | Latency |
|---|---:|---:|---:|---:|
| Frozen | | | | |
| + CE-WM energy only | | | | |
| + EFE steering only | | | | |
| + uncertainty gating only | | | | |
| Full CE-AIS | | | | |

## 8. 判断是否值得继续的标准

### 值得继续

满足任意一种：

```text
1. clean 下基本不掉点，OOD 下稳定提升；
2. clean 小幅提升，latency 可接受；
3. recovery setting 明显提升；
4. 多个不同架构模型上都有一致趋势。
```

### 需要调整

出现以下情况：

```text
1. frozen baseline 很强，但 CE-AIS clean 明显拉低成功率；
2. CE-AIS 只对一个模型有效，对其他模型无效；
3. latency 太高，无法作为 test-time 方法；
4. OOD 提升不稳定，方差很大。
```

对应调整：

- 减小 steering step；
- 减小 action clamp；
- 调低 gating lambda；
- 只在高不确定性状态启用 CE-AIS；
- 对 action chunk 只 steering 第一步或前几步；
- 改 CE-WM 训练数据，加入更接近目标模型动作分布的负样本。

## 9. 当前项目下一步建议

### 第一步：不要继续用 OpenVLA zero-shot 当主结果

保留它作为 weak transfer diagnostic 即可。

### 第二步：先接 RoboFlamingo

原因：

- CALVIN 经典；
- checkpoint 明确；
- 论文可解释性强。

要做：

1. 下载 RoboFlamingo 代码；
2. 下载 CALVIN ABC→D / ABCD→D checkpoint；
3. 按官方 README 跑官方 eval；
4. 记录官方结果；
5. 如果官方成功率正常，再写 `RoboFlamingoAdapter`。

### 第三步：再接 FLOWER VLA

原因：

- 新；
- 强；
- 代表 flow/diffusion action model。

要做：

1. 下载 FLOWER VLA 代码；
2. 下载 `flower_calvin_abcd` 或对应 split 权重；
3. 跑官方 eval；
4. 写 `FlowerVLAAdapter`；
5. 比较 frozen 与 CE-AIS。

### 第四步：用 RoboVLMs 做多模型泛化

如果前两个模型结果有趋势，再接 RoboVLMs。它适合支撑论文中的“CE-AIS is model-agnostic”。

## 10. 风险与应对

### 风险 1：模型官方代码环境冲突

应对：

- 每个外部模型单独建环境；
- 不要把依赖全部装进 CE-AIS `.venv`；
- adapter 可通过 subprocess / RPC / wrapper 脚本调用，必要时再统一。

### 风险 2：action 输出不是 `rel_actions`

应对：

- 先确认官方 env step 使用的 action 格式；
- 找官方 evaluation 中 action postprocess；
- adapter 必须输出 CALVIN env 可直接执行的 7D action。

### 风险 3：CE-AIS 对强模型 clean 反而降分

应对：

- clean 场景中只在高 uncertainty 时启用；
- 把主要贡献放到 OOD / recovery；
- 报告 gating 触发率和 energy 变化。

### 风险 4：CE-WM 和新 VLA 动作分布不匹配

应对：

- 用新 VLA 生成的 action 轨迹补充 CE-WM 训练/校准数据；
- 加入 model-specific action noise negative samples；
- 保持 VLA 参数冻结，只更新 CE-WM 或只做离线校准。

## 11. 最小可执行里程碑

### Milestone 1

```text
RoboFlamingo 官方 eval 在本地跑通，获得非零成功率。
```

### Milestone 2

```text
RoboFlamingoAdapter 跑通，frozen adapter 成功率接近官方 eval。
```

### Milestone 3

```text
RoboFlamingo + CE-AIS 在 clean 下不明显掉点。
```

### Milestone 4

```text
RoboFlamingo + CE-AIS 在 visual OOD 或 action noise 下优于 frozen。
```

### Milestone 5

```text
FLOWER VLA 重复 Milestone 1-4。
```

如果 Milestone 4 和 5 都成立，论文方向就比较稳。若只有一个模型成立，需要继续接 RoboVLMs / UniVLA 验证泛化。
