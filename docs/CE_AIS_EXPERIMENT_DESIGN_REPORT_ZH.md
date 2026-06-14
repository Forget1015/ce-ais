# CE-AIS 顶会论文实验设计报告

本文档用于统筹 CE-AIS 后续论文实验。目标不是简单列一批实验，而是围绕“顶会论文是否有说服力”来设计主实验、OOD 实验、消融实验、创新性分析、baseline 选择、数据集选择、复现实操路线和最终论文表格结构。

CE-AIS 当前定位是：面向冻结 VLA / action model 的测试时动作空间能量验证与安全校正框架。它不直接替换底层策略，而是在已有策略输出动作后，用冻结 CE-WM 的能量景观、不确定性门控、信任域约束和 accept/reject 机制进行局部动作修正。

因此实验设计必须回答以下核心问题：

1. CE-AIS 是否能作为 model-agnostic 插件作用于不同底层 action model / VLA？
2. CE-AIS 是否能在 clean setting 中不破坏强 baseline？
3. CE-AIS 是否能在 OOD、扰动、长程错误累积或 recovery setting 中带来稳定收益？
4. CE-AIS 的收益是否来自本文创新组件，而不是偶然调参？
5. CE-AIS 的额外延迟、干预率、动作偏移是否可解释且可接受？
6. 和已有 SOTA、OOD robustness、test-time adaptation、diffusion/action refinement 方法相比，CE-AIS 的新意和优势在哪里？

---

## 1. 论文实验主张

### 1.1 建议主张

不建议把主张写成：

“CE-AIS 在所有 clean / OOD 设置下都显著提升 SOTA。”

这太强，也和当前 CALVIN 结果不完全一致。

更合理、也更适合顶会的主张是：

“CE-AIS 是一个冻结模型、测试时、动作空间的可插拔能量验证与恢复框架。它能在不重新训练底层 VLA 的情况下，对多种 action model / VLA 的动作进行选择性校正；在 clean setting 中保持强 baseline，在 OOD、长程任务和 recovery setting 中提升鲁棒性，并提供可解释的干预诊断。”

可以拆成四个实验命题：

1. Clean preservation：在标准 CALVIN / LIBERO 任务上，CE-AIS 不显著破坏 strong baseline，并在部分长程任务上有小幅提升。
2. OOD robustness：在视觉、空间、物体、相机、语言 paraphrase、初始状态扰动等 setting 下，CE-AIS 提升或保持鲁棒成功率。
3. Recovery：在失败轨迹中间状态、偏离专家轨迹状态、长程任务中后段，CE-AIS 更能恢复到成功轨迹。
4. Model-agnostic：CE-AIS 不只对 FLOWER 有效，也能接到 OpenVLA / OpenVLA-OFT、Diffusion Policy、BC-Transformer、π0/π0.5 等模型输出动作上。

---

## 2. 数据集选择

顶会论文建议使用两个主数据集：

1. CALVIN：已有实验基础，适合长程语言条件机器人操作，且当前代码已经跑通 FLOWER + CE-AIS。
2. LIBERO：当前 VLA 社区非常常用，OpenVLA / π0 / diffusion / transformer baselines 更容易对齐，也适合做 spatial/object/goal/long 和 OOD/robustness 分析。

这两个数据集互补：

- CALVIN 强调长程连贯任务、ABC→D 环境泛化、历史上 L1-L5/avg length 指标成熟。
- LIBERO 强调可控任务族、标准 suite、多模型 baseline、robustness 扩展和当前 VLA 社区可复现性。

---

## 3. CALVIN 实验设计

### 3.1 为什么保留 CALVIN

CALVIN 是语言条件机器人长程操作经典 benchmark。它的特点是：

- 多个桌面环境配置，例如 A/B/C/D；
- 语言指令驱动的连续子任务；
- 常见评估是 5 个连续任务组成一个 chain；
- 指标包括 L1-L5 和 average completed length；
- 对长程错误累积非常敏感。

CE-AIS 的动作校正、能量验证和 recovery 机制天然适合 CALVIN，因为长程 chain 中，一个小动作错误可能导致后续任务失败。

### 3.2 CALVIN 推荐使用 split

建议至少使用：

1. ABC→D：训练/适配在 A/B/C，测试在 held-out D。该 split 更能体现泛化，是论文主 split。
2. ABCD→D 或 D→D：如果可复现 FLOWER 官方 checkpoint，可作为 upper-bound 或 in-distribution 对照。

当前本地主要结果是 CALVIN ABC→D 上 FLOWER local reproduction：

- frozen FLOWER avg completed length ≈ 3.98；
- CE-AIS avg completed length ≈ 4.04；
- FLOWER paper / public table 报告 ABC→D 通常在 4.53/4.54 左右，ABCD→D 在 4.67 左右。

因此论文里必须谨慎处理本地 FLOWER baseline 和官方 reported SOTA 的差距。

### 3.3 CALVIN 标准指标

设第 i 条 chain 完成连续任务数为 c_i，最大为 5。

Lk 指标：

Lk = 完成至少 k 个连续任务的 chain 比例。

平均完成长度：

AvgLen = 所有 chain 完成任务数的平均值。

建议报告：

- L1, L2, L3, L4, L5；
- avg completed tasks；
- completed tasks distribution；
- conditional success P(task k success | previous tasks succeeded)；
- per-step latency；
- CE-AIS accepted rate / rejected rate / fallback rate；
- action_delta_inf_mean；
- energy_before_mean / energy_after_mean；
- uncertainty_mean / gating_lambda_mean。

### 3.4 CALVIN baseline 调研结果

以下数字来自公开论文/项目表格调研，需要在最终论文中再次核对原文和版本。

| Method | Split | L1 | L2 | L3 | L4 | L5 | Avg Len | 仓库/可复现性 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HULC | ABCD→D | 88.9 | 73.3 | 58.7 | 47.5 | 38.3 | 3.06 | public repo，老 baseline |
| RT-1 retrained | ABCD→D | 84.4 | 61.7 | 43.8 | 32.3 | 22.7 | 2.45 | 多见于 RoboFlamingo 对比，非官方 RT-1 deployment |
| RoboFlamingo | ABCD→D | 96.4 | 89.6 | 82.4 | 74.0 | 66.0 | 4.09 | public repo |
| RoboFlamingo | ABC→D | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47/2.48 | public repo |
| GR-1 | ABCD→D | 94.9 | 89.6 | 84.4 | 78.9 | 73.1 | 4.21 | public repo |
| GR-1 | ABC→D | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 | public repo |
| MDT | ABCD→D | 98.6 | 95.8 | 91.6 | 86.2 | 80.1 | 4.52 | public repo，强 diffusion transformer baseline |
| MoDE | ABCD→D | 97.1 | 92.5 | 87.9 | 83.5 | 77.9 | 4.39 | public repo |
| FLOWER | ABCD→D | 99.1 | 97.8 | 95.2 | 92.4 | 87.8 | 4.67 | public repo/checkpoint，当前最相关 |
| FLOWER | ABC→D | 99.3 | 95.9 | 90.5 | 84.8 | 77.5 | 4.54 | public repo/checkpoint，主对比目标 |
| VPP | ABC→D | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29/4.33 | 需核查代码可复现性 |
| Seer-Large | ABC→D | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 | public repo |
| AVA-VLA | ABC→D | 待核查 | 待核查 | 待核查 | 待核查 | 待核查 | 约 4.65 | 2026 方法，需重点核查 |

注意：有些表格中 reported avg length 与 L1-L5 之和不完全一致，可能来自不同评估协议、四舍五入或表格转录问题。最终论文中引用前必须逐篇核对。

### 3.5 本地 FLOWER 复现低于论文数字的解释（已定位根因）

本地 FLOWER 在 CALVIN ABC→D 上早期实验 avg length 约 3.98，而 FLOWER 论文报告 4.54。经过逐一排查，已确认**根本原因是评估序列来源不同**。

#### 3.5.1 已排除的因素

以下因素经验证后已排除：

1. **权重加载问题**：safetensors checkpoint 包含 1036 个 keys，与模型 state_dict 完全对齐（max diff = 0）。2 个命名不一致的 keys（`vlm.language_final_logits_bias` → `vlm.language_model.final_logits_bias`，`vlm.language_shared.weight` → `vlm.language_model.model.shared.weight`）已通过 adapter remap 正确加载。其中 `final_logits_bias` 在 checkpoint 中本身为全零（T5/BART 系列正常初始化），`shared.weight` 已正确映射。

2. **EMA 权重问题**：HuggingFace 上的 safetensors checkpoint README 明确标注对应 ABC→D 4.54 的结果，说明导出时已包含最终权重。

3. **image transform 差异**：adapter 的 resize+normalize 和官方 torchvision transform pipeline 之间 max diff < 0.008，mean diff < 0.004，不影响性能。

4. **action chunking / multistep 机制**：FLOWER 模型内部 `step()` 方法自行管理 `rollout_step_counter`，每 `multistep=10` 步重新推理，中间步从缓存取。adapter 的 `chunk_size=1` 是正确用法——每次调用 `model.step()` 只返回当前步动作。

5. **rollout_step_counter reset**：在 `run_chain_evaluation` 中，`policy_reset_fn` 在每个子任务开始时调用 `model.reset()`，正确清零 counter。

6. **本地代码改动**：`resize_token_embeddings(len(self.tokenizer), mean_resizing=False)` 的改动经对比不影响 action decoder 权重加载。

#### 3.5.2 确认的根因：评估序列来源

FLOWER 论文使用 `flower/evaluation/multistep_sequences.py` 中的 `get_sequences(1000)` 函数生成评估序列。该函数特点：

- 固定 seed=0，确定性生成；
- 枚举所有可能的环境初始状态（slider、drawer、block 位置组合）；
- 从每种初始状态出发，用约束求解器生成合法 5 步任务链；
- 保证每条链的 5 个任务类别互不相同且逻辑可达；
- 最终 shuffle 后截取前 1000 条。

而我们之前使用的 `--sequence-source official`（`sample_official_eval_specs`）虽然也采样合法序列，但使用不同的随机采样策略，产生的序列分布与 FLOWER 论文的确定性序列集不同。

验证结果对比（同一 checkpoint、同一评估代码、同一环境）：

| 序列来源 | n_chains | L1 | L2 | L3 | L4 | L5 | Avg Len |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--sequence-source official`（自采样） | 20 | 90.0% | 90.0% | 80.0% | 70.0% | 70.0% | 4.0 |
| `--sequence-source flower_official`（论文序列） | 20 | 95.0% | 95.0% | 95.0% | 90.0% | 90.0% | 4.65 |

**结论：差距完全来自评估序列，而非模型或代码问题。**

#### 3.5.3 修复措施

已在 `scripts/run_paper_experiments.py` 中新增 `--sequence-source flower_official` 选项：

- 直接调用 `flower/evaluation/multistep_sequences.get_sequences(n)` 生成论文同款序列；
- 每条序列附带确定性初始环境状态（通过 `get_env_state_for_initial_condition` 转换为 robot_obs + scene_obs）；
- 支持 `--n-chains 1000` 跑完整论文规模评估。

#### 3.5.4 论文表述建议（更新）

由于根因已定位，论文中可以这样表述：

“我们使用与 FLOWER 论文完全一致的评估序列生成协议（seed=0，1000 条确定性 5 步任务链），在同一 CALVIN ABC→D 环境中评估。本地 frozen FLOWER baseline 在该协议下达到 avg length ≈ 4.5+，与官方报告一致。CE-AIS 的增量效果在该协议下进行公平比较。”

此前报告的 3.98 结果对应的是不同序列采样策略下的结果，不应与官方 4.54 直接比较。后续所有正式实验统一使用 `--sequence-source flower_official`。

#### 3.5.5 评估环境封装修复（2026-06-12 确认）

**问题**：即使使用 `flower_official` 序列，本地 frozen FLOWER 的 L1 仍只有 ~97.5%（官方 99.3%，本地复现 99.6%）。差距来自 CE-AIS 自己封装的 `CALVINWrapper`（`src/evaluation/calvin_integration.py`）与官方 `HulcWrapper`（`flower/wrappers/hulc_wrapper.py`）之间的行为差异。

**验证实验**：在完全相同的机器、Python 环境、权重和评估序列下：
- 用 CE-AIS 的 `CALVINWrapper` → L1 = 97.5%（25/1000 失败）
- 用官方 `HulcWrapper` + `model.step` → **L1 = 99.7%**（3/1000 失败）

**根因**：`CALVINWrapper` 在观测预处理链路上与官方 `HulcWrapper` 存在微妙差异（transforms 应用方式、观测格式转换等），导致模型接收到的输入与训练时分布略有偏移。

**修复措施**：`scripts/run_paper_experiments.py` 已改为当 `--vla-type flower --sequence-source flower_official` 时，直接使用官方 `HulcWrapper`（通过 `external/flower_vla_calvin` 中已验证的代码）初始化环境和执行 rollout，不再经过 `CALVINWrapper`。具体改动：

1. 新增 `_init_flower_official_env()` 函数：调用官方 `get_default_mode_and_env` 加载模型和 HulcWrapper 环境；
2. 新增 `evaluate_method_flower_official()` 函数：复现官方 rollout 逻辑（`env.get_obs()` → `model.step(obs, goal)` → `env.step(action)` → task oracle 判定）；
3. `frozen_flower` 方法直接调用 `flower_model.step()`，与官方评估完全一致；
4. `ce_ais` 方法在官方环境上运行，topology 从 HulcWrapper 格式观测中适配输入；
5. 非 flower 的 VLA 类型（如 openvla、robovlms）仍走原有 `CALVINWrapper` 路径。

**更新后的 baseline 数字**（frozen FLOWER, 1000 chains, flower_official 序列）：

| 评估环境 | L1 | 失败链数 |
|---|---|---|
| 官方 HulcWrapper（本次修复后） | **99.7%** | 3 |
| 论文报告 | 99.3% | ~7 |
| 旧 CALVINWrapper（已废弃） | 97.5% | 25 |

后续所有 FLOWER 相关实验均使用官方环境路径，确保 baseline 与论文对齐。

#### 3.5.6 完整 1000 chains（L1-L5）复现的额外修复（2026-06-13 确认）

§3.5.5 把 `run_paper_experiments.py` 切到官方 HulcWrapper 路径后，**只跑 L1 子任务**时仍出现 L1≈97.7%（远低于 §3.5.5 直接用桥接脚本 `eval_flower_official_logic.py` 验证的 99.7%）。逐项排查后定位 4 个额外的隐藏差异，全部修复后跑出与论文完全一致的 **Avg Len = 4.54**。

##### A. `--device` 选项无声覆盖了 `CUDA_VISIBLE_DEVICES`

`main()` 入口的下面这段代码会在用户传了 `--device cuda:N` 时**删掉 `CUDA_VISIBLE_DEVICES`**：

```python
if args.device and str(args.device).startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ.pop("CUDA_VISIBLE_DEVICES")
```

后果：用户 `CUDA_VISIBLE_DEVICES=6 python ... --device cuda:0` 期望跑物理 GPU6，实际跑了物理 GPU0；同时 `EGL_VISIBLE_DEVICES` 也跟着错位，导致 PyBullet 渲染来自跟模型不同的 GPU，渲染数值会有微小差异。修复：删除这段 pop 逻辑，尊重用户设置的 `CUDA_VISIBLE_DEVICES`。

##### B. 评估期间的全局 `torch.manual_seed(args.seed)` 改变了 FLOWER flow sampling 噪声

`run_paper_experiments.py` 默认 `--seed 42`，会调用 `torch.manual_seed(42)`。但官方 FLOWER eval（`eval_calvin.yaml` + `eval_shard.py`）固定用 `seed_everything(0)`。差异会传递到 FLOWER 的 rectified-flow 推理：`sample_actions()` 起点 `noise = torch.randn(...)`，不同种子会产出不同噪声、不同动作轨迹，最终少量 chain 落入失败路径。

修复：删除 `torch.manual_seed(args.seed)`；改用 `seed_everything(0, workers=True)`，并保留 `rng = np.random.RandomState(args.seed)` 仅用于非 flower_official 的序列采样。

##### C. `_init_flower_official_env` 与 `load_flower_official_eval_specs` 之间消耗了 random numbers

`seed_everything(0)` 之后还要经过模型加载（含 `resize_token_embeddings(..., mean_resizing=True)` 会消耗 `torch.randn`）和 `get_sequences(1000)` 等步骤，进入第一条 chain 时 random state 已经不在 seed=0 的起点。FLOWER 的 step 推理对 random state 敏感，导致前若干 chain 就出现 2 个失败。

修复：在进入 `for method_name in args.methods:` 评估循环前**再次** `seed_everything(0, workers=True)`，把 torch/numpy/python random state 拉回到与官方桥接脚本一致的起点。

##### D. ⚠️ 真正的根因——自写的 `get_env_state_for_initial_condition` 与官方实现行为不同

这是最隐蔽也是影响最大的一处。`scripts/run_paper_experiments.py` 第 155 行自己定义了一份 `get_env_state_for_initial_condition`，`load_flower_official_eval_specs` 第 269 行用的是这份本地函数，而**不是** `flower.evaluation.utils` 里的官方版本。两者有两个关键差异：

1. **deterministic seed 的 hash 方式不同**
   - 本地：`seed = zlib.crc32(str(tuple(initial_condition.values())).encode("utf-8"))`
   - 官方：`seed = hasher(str(initial_condition.values()))`，其中 `hasher` 是 `pyhash.fnv1_32()` 实例
   - 同一 `initial_condition` 经过两种 hash 得到完全不同的 seed → `np.random.shuffle(block_table)` 和 `np.random.uniform(rot_z_range)` 取值不同 → 桌面块的位置和朝向不一样。

2. **红/蓝/粉三色块的位置分配逻辑不同**
   - 本地用一个循环 + 简化逻辑：`block_table[1 if color == "pink_block" else 0]`
   - 官方按红→蓝→粉顺序分别处理，blue_block 还要看 red_block 是否在 table（决定走 `block_table[0]` 还是 `block_table[1]`），pink_block 总是用 `block_table[1]`

后果：本地脚本生成的 `scene_obs` 与官方评估在前述确定性 1000 条链上的**初始场景**不一致——同样的语言指令面对的物体姿态不同，FLOWER 在某些条件下抓取失败的频率显著升高，整体 L1 从 99.7% 掉到 ~97.7%。

修复：让 `load_flower_official_eval_specs` 直接调用官方实现：

```python
from flower.evaluation.utils import get_env_state_for_initial_condition as _official_get_env_state
...
robot_obs, scene_obs = _official_get_env_state(initial_state)
```

本地那份函数保留给非 flower_official 的 `sample_official_eval_specs` 使用，互不影响。

##### 修复后的最终结果

`scripts/run_paper_experiments.py` 跑 frozen_flower, 1000 chains, chain_length=5, max_steps=360, GPU6, `seed_everything(0)`：

| 指标 | 论文 (FLOWER ABC→D) | 本次复现 |
|---|---|---|
| L1 | 99.3% | **99.8%** |
| L2 | 95.9% | 97.2% |
| L3 | 90.5% | 92.3% |
| L4 | 84.8% | 86.3% |
| L5 | 77.5% | 78.8% |
| **Avg Len** | **4.54** | **4.54** |
| Latency | – | 102.5 ms |

完全对齐论文报告。后续所有 FLOWER baseline / CE-AIS 实验都在这条已验证的评估链路上运行。

##### 检查清单（迁移机器或新建评估脚本时复核）

1. ✅ `--vla-type flower --sequence-source flower_official` 走 `_init_flower_official_env` + `evaluate_method_flower_official`，不经过 `CALVINWrapper`；
2. ✅ `main()` 入口不要 pop `CUDA_VISIBLE_DEVICES`；
3. ✅ 入口和评估循环前各调一次 `seed_everything(0, workers=True)`，禁止额外的 `torch.manual_seed(args.seed)` 覆盖；
4. ✅ `load_flower_official_eval_specs` 必须 `from flower.evaluation.utils import get_env_state_for_initial_condition`，不要使用脚本里本地定义的同名函数；
5. ✅ `PYTHONPATH` 把 `external/flower_vla_calvin/calvin_env` 和 `external/flower_vla_calvin` 放在前面，确保用复现版（`play_table_env.py` 第 72 行已注释 `get_git_commit_hash` 那行）；
6. ✅ `ep_len = args.max_steps = 360`，`num_sampling_steps=4`，`multistep=10`，与官方 `eval_calvin.yaml` 一致。

### 3.6 CALVIN 主实验表格模板

#### Table C1: CALVIN clean long-horizon performance

| Method | Split | Chains | L1 | L2 | L3 | L4 | L5 | Avg Len | Latency ms | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Official FLOWER reported | ABC→D | 1000 | 99.3 | 95.9 | 90.5 | 84.8 | 77.5 | 4.54 | - | paper/reference only |
| Local frozen FLOWER (official env) | ABC→D | 1000 | **99.8** | **97.2** | **92.3** | **86.3** | **78.8** | **4.54** | 102.5 | §3.5.5 + §3.5.6 全部修复后 |
| Local FLOWER + CE-AIS (official env) | ABC→D | 1000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | HulcWrapper, §3.5.5 |
| ~~Local frozen FLOWER (old wrapper)~~ | ~~ABC→D~~ | ~~1000~~ | ~~97.5~~ | ~~91.8~~ | ~~85.5~~ | ~~79.7~~ | ~~73.6~~ | ~~4.28~~ | ~~41~~ | ~~已废弃，CALVINWrapper 有误差~~ |
| OpenVLA fine-tuned + CE-AIS | ABC→D | TBD | | | | | | | | if feasible |
| π0/π0.5 fine-tuned | ABC→D | TBD | | | | | | | | if feasible |
| π0/π0.5 + CE-AIS | ABC→D | TBD | | | | | | | | if feasible |

#### Table C2: CALVIN OOD severity sweep

| OOD Type | Severity | Frozen FLOWER Avg | CE-AIS Avg | Δ Avg | Frozen L1 | CE-AIS L1 | CE-AIS accepted rate | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| physics | mild | 0.19 | 0.16 | -0.03 | | | | current result mixed |
| physics | medium | 0.07 | 0.06 | -0.01 | | | | severe distribution shift |
| physics | severe | 0.08 | 0.09 | +0.01 | | | | stress test |
| visual | mild | 0.42 | 0.45 | +0.03 | | | | positive |
| visual | medium | 0.49 | 0.42 | -0.07 | | | | negative |
| visual | severe | 0.43 | 0.44 | +0.01 | | | | slight positive |
| camera | mild | 0.46 | 0.44 | -0.02 | | | | mixed |
| camera | medium | 0.45 | 0.44 | -0.01 | | | | mixed |
| camera | severe | 0.47 | 0.43 | -0.04 | | | | negative |

### 3.7 CALVIN 后续必要实验

建议按优先级执行：

1. 复现协议排查：用官方 FLOWER eval pipeline 跑 frozen FLOWER 1000 chains 或至少 500 chains，确认本地 baseline 是否能接近 4.5。
2. CE-AIS clean 500/1000 chains：如果算力允许，使用同一 protocol 跑 frozen vs CE-AIS。
3. OOD severity 复测：当前 100 episodes 结果可以作为 pilot，但不适合做最终强结论。
4. Failure-subset / recovery：从 frozen FLOWER 失败轨迹中抽中间状态，让 CE-AIS 评估 recovery，这是更适合 CE-AIS 的创新性实验。
5. Latency optimization ablation：展示 mc_samples、n_steps、grad_mode 对 latency/performance 的影响。

---

## 4. LIBERO 实验设计

### 4.1 为什么加入 LIBERO

LIBERO 是当前 VLA / robot imitation 社区非常常用的 benchmark。它的优势是：

- suite 划分清楚；
- OpenVLA、OpenVLA-OFT、π0/π0.5、Diffusion Policy、BC-Transformer 等 baseline 更容易对齐；
- 任务短程和长程都有；
- 适合做 spatial/object/goal/language/generalization/OOD 分析；
- 相比 CALVIN，更容易展示 model-agnostic。

如果论文只用 CALVIN + FLOWER，审稿人可能认为方法只对一个模型/一个数据集有效。加入 LIBERO 后，可以支撑“通用测试时动作校正框架”的说法。

### 4.2 LIBERO suite 推荐

建议主实验使用四个 standard suites：

1. LIBERO-Spatial
2. LIBERO-Object
3. LIBERO-Goal
4. LIBERO-Long / LIBERO-10

如果算力允许，再加入：

5. LIBERO-90

各 suite 作用：

| Suite | 是否主实验 | 作用 | 为什么适合 CE-AIS |
|---|---|---|---|
| LIBERO-Spatial | 必选 | 空间关系泛化 | 动作校正可修复空间偏差 |
| LIBERO-Object | 必选 | 物体识别/操作 | 检验是否只依赖空间先验 |
| LIBERO-Goal | 必选 | 目标条件任务 | 检验语言目标和动作能量一致性 |
| LIBERO-Long / LIBERO-10 | 必选 | 长程任务 | 最适合展示 recovery 和误差累积修复 |
| LIBERO-90 | 强烈建议 | 规模泛化 | 提升顶会说服力，但计算更重 |

### 4.3 LIBERO 主指标

标准指标：

- per-task success rate；
- per-suite average success；
- overall average success；
- trials per task，例如 10/20/50 rollouts。

CE-AIS 额外指标：

- intervention rate；
- accepted/rejected/abstained/fallback rate；
- action deviation mean/max；
- energy_before/after；
- latency overhead；
- recovery success；
- robust success under perturbation。

### 4.4 LIBERO baseline 推荐

#### Tier 1：必须跑

1. Base policy without CE-AIS

这是所有 CE-AIS 实验的核心。每个底层模型都必须有：

- model only；
- model + CE-AIS。

否则无法证明 CE-AIS 的插件效果。

2. OpenVLA / OpenVLA-OFT

OpenVLA 是当前最重要的开源 VLA baseline 之一。OpenVLA-OFT 在 LIBERO 上尤其相关，因为它有面向机器人控制 fine-tuning/eval 的实现。

推荐：

- OpenVLA-OFT on LIBERO；
- OpenVLA-OFT + CE-AIS。

这是 LIBERO 主实验最关键的一组。

3. Diffusion Policy

Diffusion Policy 是 action model 领域最经典 baseline。CE-AIS 是 action-space steering，接到 diffusion policy 的 action chunk 上逻辑非常自然。

推荐：

- Diffusion Policy；
- Diffusion Policy + CE-AIS；
- 可选：Diffusion Policy + low-energy rerank。

4. BC-Transformer / ResNet-Transformer

作为传统 imitation/action transformer baseline，便于说明 CE-AIS 不依赖 VLA 大模型，也能用于普通 action policy。

推荐：

- BC-Transformer；
- BC-Transformer + CE-AIS。

#### Tier 2：强烈建议跑

5. π0 / π0.5 via OpenPI

π0/π0.5 是目前非常新的强 VLA/action foundation model。若能在 LIBERO 上跑通，论文说服力会大幅提升。

推荐优先级：

- π0.5-LIBERO checkpoint 或 OpenPI LIBERO config；
- π0/π0.5 + CE-AIS。

注意：需要确认 checkpoint、license、LIBERO 版本和 evaluation protocol。

6. Octo

Octo 是开源 generalist robot policy。若能跑通，可作为 foundation-action-model 对比。但它不是第一优先级，因为接入成本可能高于 OpenVLA/π0。

#### Tier 3：引用为 related work，不建议作为主 runnable baseline

1. RT-1 / RT-2

重要但不适合作为主要 LIBERO 可复现 baseline，除非有明确、可信、公开的 LIBERO fine-tuned checkpoint。

2. RT-X / RT-1-X / RT-2-X

同理，更适合 related work 背景。

### 4.5 LIBERO 主实验表格模板

#### Table L1: Standard LIBERO success

| Method | Spatial | Object | Goal | Long | Avg | Trials/task | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| BC-Transformer | | | | | | | |
| BC-Transformer + CE-AIS | | | | | | | |
| Diffusion Policy | | | | | | | |
| Diffusion Policy + CE-AIS | | | | | | | |
| OpenVLA-OFT | | | | | | | main VLA baseline |
| OpenVLA-OFT + CE-AIS | | | | | | | main result |
| π0/π0.5 | | | | | | | if feasible |
| π0/π0.5 + CE-AIS | | | | | | | if feasible |
| Octo | | | | | | | optional |
| Octo + CE-AIS | | | | | | | optional |

#### Table L2: LIBERO-90 generalization

| Method | LIBERO-90 Avg Success | Std/CI | Trials/task | Latency | Notes |
|---|---:|---:|---:|---:|---|
| OpenVLA-OFT | | | | | |
| OpenVLA-OFT + CE-AIS | | | | | |
| π0/π0.5 | | | | | |
| π0/π0.5 + CE-AIS | | | | | |

---

## 5. OOD / Robustness 实验设计

CE-AIS 的创新点最适合在 OOD、recovery、long-horizon robustness 中体现。标准 clean success 不够，因为 strong VLA 已经很强，CE-AIS 的主要价值是“发现动作可能不可靠并局部修正”。

### 5.1 OOD 类型

建议在 CALVIN 和 LIBERO 中都做以下 OOD 类型。

#### 视觉 OOD

- brightness / contrast 改变；
- Gaussian noise；
- background texture 改变；
- camera shift；
- partial occlusion；
- distractor objects。

#### 空间 OOD

- 目标位置偏移；
- 物体初始位姿扰动；
- receptacle / container 位置变化；
- robot initial pose perturbation。

#### 物体 OOD

- distractor objects；
- similar-looking object；
- object pose/scale variation；
- unseen object instance if benchmark supports。

#### 语言 OOD

- paraphrase；
- synonym replacement；
- instruction reorder；
- longer natural-language command；
- underspecified command。

#### Dynamics / physics OOD

CALVIN 已有：

- mass scale；
- friction scale。

LIBERO/robosuite 中可以考虑：

- object mass perturbation；
- friction perturbation；
- control noise；
- action delay。

注意：严重 physics OOD 可能让 VLA 和 CE-WM 都失效，不建议作为主成功 claim，应作为 stress test。

### 5.2 OOD severity 设计

建议统一使用 mild / medium / severe 三档。

示例：

| OOD Type | Mild | Medium | Severe |
|---|---|---|---|
| visual brightness | 0.8 | 0.6 | 0.5 |
| visual noise std | 0.02 | 0.05 | 0.10 |
| camera offset | 0.01 | 0.03 | 0.05 |
| physics mass scale | 1.2 | 1.5 | 2.0 |
| physics friction scale | 0.8 | 0.5 | 0.3 |
| object pose | small | medium | large |
| language paraphrase | simple synonym | sentence rewrite | long/ambiguous rewrite |

### 5.3 Robustness 表格模板

#### Table R1: CALVIN OOD robustness

| Method | Physics Mild | Physics Med | Physics Severe | Visual Mild | Visual Med | Visual Severe | Camera Mild | Camera Med | Camera Severe | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen FLOWER | | | | | | | | | | |
| FLOWER + CE-AIS | | | | | | | | | | |
| FLOWER + CE-AIS fast | | | | | | | | | | |

#### Table R2: LIBERO robustness

| Method | Visual Perturb | Spatial Perturb | Object Distractor | Paraphrase | Recovery State | Avg Robust Success |
|---|---:|---:|---:|---:|---:|---:|
| OpenVLA-OFT | | | | | | |
| OpenVLA-OFT + CE-AIS | | | | | | |
| π0/π0.5 | | | | | | |
| π0/π0.5 + CE-AIS | | | | | | |
| Diffusion Policy | | | | | | |
| Diffusion Policy + CE-AIS | | | | | | |

---

## 6. Recovery 实验设计

Recovery 是 CE-AIS 最重要的创新性实验之一。因为 CE-AIS 本质是动作验证与局部恢复控制器，而不是重新训练策略。

### 6.1 Failure-state recovery

做法：

1. 用 frozen baseline 跑若干 rollouts。
2. 保存失败轨迹中的中间状态。
3. 从这些失败前/失败中状态重新初始化环境。
4. 对比 baseline 和 baseline + CE-AIS 的 recovery success。

指标：

- recovery success rate；
- time-to-recover；
- subsequent task completion；
- CE-AIS intervention rate during recovery；
- energy decrease during successful recovery。

表格模板：

| Dataset | Base Policy | Start State Type | Base Recovery | +CE-AIS Recovery | Δ | Intervention Rate | Notes |
|---|---|---|---:|---:|---:|---:|---|
| CALVIN | FLOWER | pre-failure | | | | | |
| CALVIN | FLOWER | failed intermediate | | | | | |
| LIBERO-Long | OpenVLA-OFT | off-nominal | | | | | |
| LIBERO-Long | π0/π0.5 | off-nominal | | | | | |

### 6.2 Long-horizon partial-progress recovery

对于长程任务，完整 success 可能太稀疏。建议报告 stage-level 或 subtask-level partial progress。

指标：

- completed subgoals；
- conditional success；
- failure position distribution；
- CE-AIS 是否减少 early failure。

---

## 7. 与本文创新点对应的实验

CE-AIS 的创新点可以拆成以下组件：

1. frozen VLA + frozen CE-WM 的测试时动作空间控制；
2. CE-WM 能量 verifier；
3. MC-Dropout uncertainty gating；
4. trust-region action steering；
5. accept/reject safety layer；
6. calibrated NCE CE-WM training；
7. model-agnostic adapter。

实验必须分别证明这些组件有必要。

---
## 7.1 CE-AIS 组件理解与常见问题专栏

本节用于解释 CE-AIS 在实验设计中被拆分为多个组件的原因，以及每个组件在方法、实验和论文创新点中的作用。CE-AIS 并不是一个单一的动作预测模型，而是一个“冻结 VLA 主策略 + 测试时动作空间校正 + 安全接受/拒绝”的完整推理框架。因此，在论文实验中需要分别验证每个组件是否真正有效。

### Q1: 为什么 CE-AIS 要拆成多个组件做实验？

因为 CE-AIS 的目标不是重新训练一个更强的机器人策略，而是在已有 VLA 或 action model 的基础上，作为一个测试时插件，对原始动作进行安全、局部、可拒绝的修正。

如果只报告“CE-AIS 整体效果比 baseline 好”，审稿人可能会问：

- 提升到底来自 CE-WM，还是来自随机扰动？
- 安全机制是否真的有用？
- 不确定性门控是否真的有必要？
- 如果不做 trust region，结果会不会更差？
- 如果没有 accept/reject，CE-AIS 是否会破坏强 VLA 的 clean 性能？
- 这个方法是否只对 FLOWER 有效，还是能迁移到 OpenVLA、π0.5 等其他模型？

因此，实验设计需要把 CE-AIS 拆成不同组件，通过消融实验逐一证明每个设计的必要性。

---

### Q2: VLA prior 是什么意思？

VLA prior 指的是原始视觉-语言-动作模型输出的动作先验。

在本文框架中，FLOWER、OpenVLA、π0.5 等模型都可以作为 VLA prior。它们接收当前观测和语言指令，输出原始动作：

` ` `text
a0 = VLA(observation, instruction)
` ` `


其中 `a0` 是原始动作，也就是 CE-AIS 进行修正之前的动作。

CE-AIS 不替代 VLA，而是在 VLA 输出动作之后，对其进行小幅校正。因此 VLA prior 是整个系统的主策略，CE-AIS 是测试时的安全辅助模块。

对应实验目标：
证明 CE-AIS 不是只依赖某一个特定 VLA，而是可以作为 plug-in 模块接入不同 backbone。

推荐实验：
- FLOWER + CE-AIS;
- OpenVLA / OpenVLA-OFT + CE-AIS;
- π0 / π0.5 + CE-AIS;
- Diffusion Policy 或 BC-Transformer + CE-AIS。

如果 CE-AIS 在多个 backbone 上都有一致收益，就能支持“model-agnostic test-time action steering”的创新点。

### Q3: CE-WM 是什么意思？

CE-WM 是 Causal Energy World Model，也就是因果能量世界模型。

它的作用不是直接输出动作，而是评价一个状态-动作序列是否合理。

简单理解：
- 专家动作应该有低能量；
- 扰动动作、错误动作、危险动作应该有高能量。

所以 CE-WM 相当于 CE-AIS 的“裁判”或“副驾驶”。

VLA 提出动作：
`a0`

CE-WM 判断这个动作是否合理：
`E(z, a0)`

如果能量高，说明这个动作可能不太符合专家轨迹或任务动力学；如果能量低，说明动作更可能合理。

对应实验目标：
证明动作校正不是随机扰动，而是由学习到的能量模型引导。

推荐消融实验：
- CE-AIS without CE-WM：去掉 CE-WM，只使用 VLA 原始动作；
- Random energy：用随机能量替代 CE-WM；
- MLP energy：用简单 MLP 替代 CE-WM；
- uncalibrated CE-WM：使用未校准的 NCE CE-WM；
- calibrated CE-WM：使用当前校准后的 CE-WM。

如果 calibrated CE-WM 表现更稳，说明因果能量世界模型确实是有效组件。

---

### Q4: Uncertainty estimation 是什么意思？

Uncertainty estimation 指的是 CE-WM 对自己判断的“不确定性估计”。

CE-WM 可能在某些场景下不可靠，例如：
- 视觉扰动严重；
- 相机位置偏移；
- 物理参数变化；
- 当前状态远离训练分布；
- VLA 输出了非常奇怪的动作。

如果 CE-WM 自己也不确定，那么 CE-AIS 不应该盲目相信它去修改 VLA 动作。

当前实现使用 MC-Dropout 估计不确定性。也就是说，同一个输入让 CE-WM 多次 forward，每次 dropout mask 不同。如果多次输出的能量差异很大，说明 CE-WM 不确定性高；如果多次输出很接近，说明 CE-WM 判断稳定。

对应实验目标：
证明 CE-AIS 不是每一步都强行干预，而是可以根据 CE-WM 的可靠性自适应调整。

推荐消融实验：
- without uncertainty：不使用不确定性估计；
- MC-Dropout uncertainty：当前方法；
- fixed uncertainty threshold：设置硬阈值；
- no abstention：即使不确定性高也强行干预。

如果不确定性机制能降低 clean 退化，并提升 OOD 下稳定性，就能支持“abstentive test-time control”的创新点。

---

### Q5: Bilateral gating 是什么意思？

Bilateral gating 可以翻译成“双向门控”。

它的作用是根据不确定性决定 CE-AIS 的干预强度。

门控输出一个数：
`lambda`

这个 `lambda` 可以理解成 CE-AIS 的“干预油门”：
- lambda 大：CE-AIS 更积极地修改动作；
- lambda 小：CE-AIS 少改动作；
- lambda 接近 0：几乎不干预，更多相信 VLA 原始动作。

为什么叫“双向”？

因为它不是简单地认为“不确定性越低越好”，而是认为当前不确定性应该接近历史稳定水平。

如果当前不确定性远高于历史平均值，说明模型可能遇到了 OOD，应该少干预。

如果当前不确定性远低于历史平均值，也可能说明模型进入异常过度自信区域，也应保守。

只有当当前不确定性接近历史正常区间时，CE-AIS 才更愿意介入。

对应实验目标：
证明双向门控比固定干预或简单单向门控更加稳定。

推荐消融实验：
- fixed lambda：固定干预强度；
- no gating：不使用门控；
- one-sided gating：只在不确定性高时降低干预；
- bilateral gating：当前双向门控。

如果 bilateral gating 在 clean preservation 和 OOD robustness 上更稳，就能证明该设计有效。

---

### Q6: Steering 是什么意思？

Steering 可以翻译成“动作引导”、“动作偏转”或“动作校正”。

在 CE-AIS 中，Steering 指的是：

VLA 已经输出了原始动作 `a0`，CE-AIS 根据 CE-WM 的能量方向，在 `a0` 附近做小幅修改，得到候选动作 `a*`。

所以 CE-AIS 不是重新生成动作，而是在 VLA 原始动作附近进行局部修正。

可以理解成：
- VLA 是主驾驶；
- CE-WM 是副驾驶；
- Steering 是副驾驶轻轻帮主驾驶修方向盘。

当前实现主要使用退火 Langevin 动力学或 EFE-inspired 动作更新。它根据 CE-WM 的能量梯度，把动作往低能量方向移动一点。

对应实验目标：
证明动作空间 test-time steering 本身是否有效。

推荐消融实验：
- no steering：只评价动作，不修改动作；
- Langevin steering：当前主要方法；
- finite-difference gradient：当前默认有限差分；
- autograd gradient：用反向传播求动作梯度；
- reranking：采样多个候选动作，只选择能量最低者。

如果 steering 能在不破坏 clean 性能的前提下提升 OOD 或 recovery 性能，就能支持“test-time action-space optimization”的创新点。

---

### Q7: Trust region 是什么意思？

Trust region 可以翻译成“信任域”。

它的作用是限制 CE-AIS 不能把 VLA 原始动作改得太远。

因为 FLOWER、OpenVLA、π0.5 这些模型本身已经很强，如果 CE-AIS 修改太大，很容易把原本正确的动作改坏。

所以 trust region 要求：
`a*` 必须在 `a0` 附近

也就是说，CE-AIS 只能做局部小修正，不能完全覆盖 VLA 的决策。

当前 `safe_balanced` 配置中，`action_delta_max = 0.08`，表示每个动作维度相对原始动作最多只能偏移 0.08。

对应实验目标：
证明 CE-AIS 是安全局部校正，而不是粗暴替换 VLA。

推荐消融实验：
- no trust region;
- delta = 0.02;
- delta = 0.05;
- delta = 0.08;
- delta = 0.10;
- delta = 0.20。

如果没有 trust region 时 clean 性能下降，而适中 trust region 能保持 clean 并提升部分 OOD，就说明该机制有效。

---

### Q8: Accept/reject 是什么意思？

Accept/reject 是动作校正后的安全检查机制。

CE-AIS 生成候选动作 `a*` 后，不会直接执行它，而是比较：
- 原始动作 `a0` 的能量；
- 修正动作 `a*` 的能量。

如果修正后能量更低，并且动作没有异常，也没有超过 trust region，则接受该动作：
`a_exec = a*`

如果修正后能量没有降低，或者动作异常，则拒绝修改，回退到原始 VLA 动作：
`a_exec = a0`

所以 accept/reject 是 CE-AIS 的最后一道安全保险。

它保证 CE-AIS 不会盲目执行所有修改，而是只有在能量模型认为修改确实更优时才采纳。

对应实验目标：
证明 CE-AIS 的安全机制可以防止 clean 性能被破坏。

推荐消融实验：
- without accept/reject：所有 steering 动作都执行；
- with accept/reject：当前方法；
- different accept margin：改变能量改善阈值。

如果去掉 accept/reject 后 clean 成功率下降，就说明该安全机制必要。

---

### Q9: Diagnostics 是什么意思？

Diagnostics 指的是 CE-AIS 的干预诊断指标。

它不是控制模块，而是论文分析中非常重要的可解释性工具。

CE-AIS 会记录：
- 总步数；
- accepted rate：有多少步接受了 CE-AIS 修改；
- rejected rate：有多少步拒绝了 CE-AIS 修改；
- abstained uncertainty rate：有多少步因为不确定性高而放弃干预；
- action delta：动作平均改变量；
- energy before：修改前能量；
- energy after：修改后能量；
- uncertainty：平均不确定性；
- gating lambda：平均干预强度。

这些指标可以解释 CE-AIS 为什么成功或失败。

例如：
如果某个 OOD 场景 CE-AIS 没有提升，可以查看：
- 是否 accepted rate 太低；
- 是否 uncertainty 太高；
- 是否 action delta 太小；
- 是否 energy after 没有明显降低；
- 是否 gating lambda 长期很低。

对应实验目标：
证明 CE-AIS 是选择性干预，而不是黑箱扰动。

推荐在所有主实验、OOD 实验和 recovery 实验中都报告 diagnostics。

---

### Q10: 这些组件和论文创新点如何对应？

CE-AIS 的论文创新点可以拆成以下几条：

第一，冻结 VLA 的测试时动作校正框架。
对应组件：
- VLA prior;
- action-space steering;
- model-agnostic adapter。

第二，基于因果能量世界模型的动作评价。
对应组件：
- CE-WM;
- calibrated NCE;
- positive/negative action energy separation。

第三，不确定性感知的自适应干预。
对应组件：
- MC-Dropout uncertainty;
- bilateral gating;
- uncertainty abstention。

第四，安全局部控制。
对应组件：
- trust region;
- accept/reject;
- fallback to VLA prior。

第五，可解释的干预分析。
对应组件：
- diagnostics;
- intervention rate;
- energy before/after;
- action delta;
- accepted/rejected statistics。

因此，实验设计中不能只报告最终成功率，还需要用消融实验和诊断指标证明每个创新点都发挥了作用。

---

### Q11: 最简单的整体理解是什么？

可以用一个“主驾驶/副驾驶”的比喻理解 CE-AIS：

- VLA 是主驾驶，负责根据图像和语言指令给出动作；
- CE-WM 是副驾驶，负责判断这个动作是否合理；
- Uncertainty 是副驾驶对自己判断的自信程度；
- Bilateral gating 是副驾驶说话声音的大小；
- Steering 是副驾驶轻轻帮主驾驶修方向盘；
- Trust region 是限制副驾驶不能猛打方向盘；
- Accept/reject 是如果副驾驶修得不好，就不采纳它的建议；
- Diagnostics 是记录副驾驶什么时候帮了忙、什么时候没帮忙、帮得是否有效。

所以 CE-AIS 的整体逻辑是：
- 不是替代 VLA，而是在测试时安全地辅助 VLA；
- 不是每一步都强行修改，而是选择性干预；
- 不是重新训练主策略，而是在冻结所有模型参数，只在动作空间做局部修正；
- 不是黑箱提升，而是可以通过 intervention diagnostics 分析干预行为。

---

### Q12: 为什么这些实验对顶会论文重要？

顶会论文通常不仅要求方法有效，还要求证据链完整。

对于 CE-AIS，审稿人可能关注：
1. 是否真的比强 VLA baseline 有收益；
2. 是否只在弱 baseline 上有效；
3. 是否能泛化到多个数据集；
4. 是否能泛化到多个 VLA backbone；
5. 是否在 OOD、recovery、long-horizon 场景下更有优势；
6. 是否会破坏 clean performance；
7. 每个模块是否有必要；
8. 额外延迟是否可接受；
9. 是否可复现。

因此，对 CE-AIS 的实验应包括：
- 主实验：CALVIN 和 LIBERO 上与强 baseline 对比；
- 多 backbone 实验：FLOWER、OpenVLA、π0.5 等；
- OOD 实验：视觉、相机、物理扰动；
- recovery 实验：失败状态恢复；
- 消融实验：去掉 CE-WM、gating、trust region、accept/reject；
- 效率实验：延迟、CE-WM forward 次数、干预频率；
- 诊断实验：accepted rate、energy decrease、action delta、不确定性。

这样才能形成完整论文逻辑：
CE-AIS 不只是一个提升成功率的技巧，而是一个面向冻结 VLA 的安全、可解释、可迁移的测试时动作校正框架。
---

## 8. 消融实验设计

### 8.1 安全组件消融

| Variant | Trust Region | Accept/Reject | Uncertainty Gate | Expected Question |
|---|---|---|---|---|
| Full CE-AIS | yes | yes | yes | 完整方法 |
| No trust region | no | yes | yes | 是否会过度偏离 VLA |
| No accept/reject | yes | no | yes | 是否会错误接受坏 steering |
| No uncertainty gate | yes | yes | no | 不确定性是否必要 |
| Always steer | no/yes | no | no | 证明 naive steering 不安全 |
| VLA only | no | no | no | base policy |

表格：

| Variant | Clean Avg | OOD Avg | Recovery Success | Action Delta | Accepted Rate | Latency | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| VLA only | | | | 0 | 0 | | |
| Always steer | | | | | 100 | | likely unsafe |
| No trust region | | | | | | | |
| No accept/reject | | | | | | | |
| No uncertainty gate | | | | | | | |
| Full CE-AIS | | | | | | | |

### 8.2 CE-WM training loss 消融

| Variant | Energy Reg | Upper Margin | Lower Margin | Expected Question |
|---|---|---|---|---|
| Raw NCE | no | no | no | 是否出现 margin 爆炸/塌缩 |
| + energy reg | yes | no | no | 控制绝对能量是否有效 |
| + margin upper | yes | yes | no | 防止 margin 过大是否有效 |
| + margin lower | yes | yes | yes | 完整 calibrated NCE |

报告指标：

- CE-WM loss curve；
- pos_energy_mean；
- neg_energy_mean；
- energy_margin；
- energy_abs_mean；
- action gradient norm；
- downstream CE-AIS success；
- downstream action_delta；
- downstream rejected rate。

表格：

| CE-WM Loss | Margin Final | Energy Abs | Grad Norm | Clean Avg | OOD Avg | Recovery | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Raw NCE | | | | | | | |
| + Energy Reg | | | | | | | |
| + Upper Margin | | | | | | | |
| Full Calibrated | | | | | | | |

### 8.3 Steering 超参数消融

| Parameter | Values | Purpose |
|---|---|---|
| n_steps | 0, 1, 2, 3 | 控制优化步数与延迟 |
| action_delta_max | 0.02, 0.05, 0.08, 0.10 | 信任域大小 |
| mc_samples | 1, 2, 5 | 不确定性质量与延迟 |
| lambda_max | 0.1, 0.2, 0.4, 0.6 | 最大干预强度 |
| accept_energy_margin | 0, 1e-4, 1e-3 | 接受严格程度 |
| grad_mode | finite_diff, autograd | 延迟与稳定性 |

表格：

| Setting | Clean Avg | OOD Avg | Recovery | Latency | Action Delta | Accept Rate |
|---|---:|---:|---:|---:|---:|---:|
| n_steps=0 | | | | | | |
| n_steps=1 | | | | | | |
| n_steps=2 | | | | | | |
| n_steps=3 | | | | | | |

### 8.4 Model-agnostic 消融

核心问题：CE-AIS 是否只对 FLOWER 有效？

表格：

| Base Model | Dataset | Base Success | +CE-AIS Success | Δ | Base Latency | +CE-AIS Latency | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| FLOWER | CALVIN | | | | | | |
| OpenVLA-OFT | LIBERO | | | | | | |
| π0/π0.5 | LIBERO | | | | | | |
| Diffusion Policy | LIBERO | | | | | | |
| BC-Transformer | LIBERO | | | | | | |

---

## 9. 对比已有 OOD / test-time 方法

### 9.1 应该比较的方法类别

CE-AIS 不是普通 VLA 训练方法，而是 test-time action steering。因此除了比较 VLA SOTA，还应比较以下类别：

1. uncertainty-based action selection；
2. action voting / perturbation learning；
3. adaptive test-time compute；
4. diffusion action reranking/refinement；
5. recovery policy / recovery optimization；
6. test-time adaptation methods。

### 9.2 相关方法候选

以下方法适合作为 related work 或 runnable baseline，具体是否能跑取决于代码/环境。

| 方法 | 类型 | 与 CE-AIS 关系 | 建议 |
|---|---|---|---|
| LIBERO-X | robustness benchmark | OOD benchmark，不一定是方法 | 可用于 robustness eval |
| LIBERO-Para | paraphrase benchmark | 语言鲁棒性 | 可用于 language OOD |
| Test-Time Perturbation Learning | test-time VLA perturb/action voting | 与 CE-AIS 很接近 | 尽量比较或复现简化版 |
| VLA-ATTC | adaptive test-time compute | uncertainty/critic 分配算力 | 可作为 related/test-time baseline |
| EVOLVE-VLA | environment feedback test-time training | 比 CE-AIS 更重，需要反馈 | related 或可比 baseline |
| HELM / LIBERO-Recovery | long-horizon memory/recovery | recovery 相关 | related，若有代码可比较 |
| RePO-VLA | recovery-driven optimization | recovery policy optimization | related |
| ADPro | test-time adaptive diffusion policy | diffusion policy adaptation | related or baseline |
| RESample | diffusion/policy resampling | action sampling/refinement | related |

### 9.3 简化但可复现的 test-time baseline

如果上述方法代码不可复现，建议实现几个简单强 baseline：

1. Random perturb + energy rerank

采样 K 个 VLA 动作附近候选，用 CE-WM 选最低能量，不做 Langevin。

2. Uncertainty abstention only

只用不确定性决定是否执行 VLA，不改动作。

3. Action smoothing / low-pass filter

简单动作平滑 baseline，排除“只是动作变平滑”的可能。

4. Multiple candidate voting

对 VLA 或 diffusion policy 采样多个动作，使用动作均值/投票/低方差选择。

表格：

| Test-Time Method | Uses CE-WM | Uses Gradient | Uses Trust Region | Clean | OOD | Recovery | Latency |
|---|---|---|---|---:|---:|---:|---:|
| VLA only | no | no | no | | | | |
| Action smoothing | no | no | no | | | | |
| Random rerank | yes | no | yes | | | | |
| Uncertainty abstain | yes | no | no | | | | |
| Candidate voting | no/yes | no | no | | | | |
| CE-AIS full | yes | finite-diff/autograd | yes | | | | |

---

## 10. Latency 与效率实验

当前 CE-AIS 延迟明显高于 frozen FLOWER，主要来自：

- MC-Dropout 多次 CE-WM forward；
- finite-difference 每个动作维度正负扰动 forward；
- Langevin 多步更新；
- E_after accept/reject 额外 forward。

必须在论文中诚实报告，并给出效率优化版本。

### 10.1 成本拆解

以当前 safe_balanced 为例：

- E_before：1 次 CE-WM forward；
- uncertainty：mc_samples=5，即 5 次 forward；
- finite_diff：动作维度 7，正负差分，n_steps=2，大约 2×7×2=28 次 forward；
- E_after：1 次 forward；
- 总计约 35 次 CE-WM forward，不含 FLOWER。

### 10.2 效率实验表格

| Variant | mc_samples | n_steps | grad_mode | CE-WM fwds approx | Success | OOD | Latency | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Full safe | 5 | 2 | finite_diff | 35 | | | | current |
| Fast uncertainty | 2 | 2 | finite_diff | 32 | | | | |
| Fast steering | 5 | 1 | finite_diff | 21 | | | | |
| Autograd | 5 | 2 | autograd | lower | | | | test stability |
| Rerank K=8 | 1/2 | 0 | no gradient | batched | | | | candidate selection |
| No uncertainty | 0/1 | 2 | finite_diff | 30 | | | | ablation |

顶会论文中建议包含一张 “Accuracy vs Latency” Pareto 图。

---

## 11. 最终论文建议实验结构

### Main Paper 主实验

建议主文放 4-5 张核心表/图。

#### Table 1: CALVIN clean long-horizon

展示 FLOWER vs FLOWER+CE-AIS，在同一 local protocol 下 clean preservation。

#### Table 2: LIBERO standard benchmark

展示 OpenVLA-OFT、π0/π0.5、Diffusion Policy、BC-Transformer 及其 +CE-AIS。

#### Table 3: OOD robustness

展示 CALVIN OOD + LIBERO robustness 平均结果。

#### Table 4: Recovery / long-horizon failure correction

展示 failure-state recovery 或 LIBERO-Long partial progress。

#### Figure 1: CE-AIS behavior diagnostics

展示 accepted/rejected rate、energy before/after、action_delta、uncertainty、latency。

#### Figure 2: Ablation/Pareto

展示 trust-region / accept-reject / uncertainty gate / CE-WM calibrated loss 的消融，以及 latency-success tradeoff。

### Appendix 附录实验

1. 更多 CALVIN L1-L5；
2. per-task LIBERO results；
3. 所有 OOD severity；
4. 更多 checkpoint selection；
5. CE-WM training curves；
6. hyperparameter sweeps；
7. failure cases qualitative videos；
8. official baseline reproduction caveats。

---

## 12. 可复现性计划

顶会论文必须强调 reproducibility。

### 12.1 代码/环境

每个数据集必须记录：

- repo commit hash；
- checkpoint path/hash；
- dataset version；
- evaluation seed；
- number of chains/episodes；
- GPU 型号；
- action chunk length；
- max steps；
- camera/rendering setting；
- success oracle version。

### 12.2 Baseline 复现原则

1. 若有官方 evaluation script，优先使用官方脚本。
2. 若使用本项目 adapter，则必须报告 adapter 与官方 protocol 的差异。
3. 对每个 baseline，先复现官方 reported number 或给出差距解释。
4. CE-AIS 与 base policy 必须使用完全相同 evaluation pipeline。
5. 不允许把官方 reported baseline 和本地 CE-AIS 直接当作严格 apples-to-apples 比较。

### 12.3 统计显著性

建议：

- CALVIN final clean 至少 500 chains，最好 1000 chains；
- LIBERO 每 task 至少 20 rollouts，最终最好 50 rollouts；
- 报告 mean ± std 或 bootstrap confidence interval；
- 对主要 gain 做 paired evaluation，如果能固定 initial states。

---

## 13. 实验优先级路线图

### Phase 0：确认 CALVIN/FLOWER 复现差距

目标：搞清楚本地 FLOWER 3.98 vs paper 4.54 的原因。

任务：

1. 用官方 FLOWER eval script 跑 frozen FLOWER；
2. 检查 checkpoint 是否为 best/EMA；
3. 检查 100 vs 1000 chains；
4. 检查 split；
5. 检查 missing/unexpected keys；
6. 记录最终可复现 baseline。

产出：一段论文/附录里可解释的 baseline reproduction note。

### Phase 1：CALVIN final experiments

目标：稳住当前已有主线。

实验：

1. frozen FLOWER vs FLOWER+CE-AIS clean；
2. OOD severity sweep；
3. safe components ablation；
4. CE-WM loss/checkpoint ablation；
5. latency Pareto。

### Phase 2：LIBERO 接入 OpenVLA-OFT

目标：建立第二数据集主结果。

实验：

1. OpenVLA-OFT baseline；
2. OpenVLA-OFT + CE-AIS；
3. Spatial/Object/Goal/Long；
4. robustness perturbations。

### Phase 3：LIBERO 多模型验证

目标：支撑 model-agnostic claim。

实验：

1. Diffusion Policy + CE-AIS；
2. BC-Transformer + CE-AIS；
3. π0/π0.5 + CE-AIS if feasible；
4. optional Octo。

### Phase 4：Recovery / innovation experiments

目标：形成论文亮点。

实验：

1. failure-state recovery；
2. long-horizon partial progress；
3. intervention diagnostics；
4. qualitative rollout videos；
5. energy landscape visualization。

---

## 14. 风险与应对

### 风险 1：CE-AIS 在 clean 上提升不大

应对：主张 clean preservation，而不是 clean SOTA。强调 CE-AIS 对 strong VLA 的安全附加层，不是替换训练策略。

### 风险 2：OOD 结果 mixed

应对：按 OOD 类型分析。physics severe 可作为 stress test；重点放在 visual/spatial/recovery/long-horizon where action correction is meaningful。

### 风险 3：延迟过高

应对：做 fast CE-AIS variant，包括 mc_samples=1/2、n_steps=1、autograd gradient、rerank batched candidate。

### 风险 4：baseline 复现不到官方 SOTA

应对：明确区分 official reported vs local reproduced。主实验只做 same-pipeline comparison。

### 风险 5：π0/π0.5 接入困难

应对：OpenVLA-OFT + Diffusion Policy + BC-Transformer 已经足够支撑第一版 model-agnostic；π0/π0.5 作为 high-impact optional。

### 风险 6：CE-WM 只适配一个底层模型

应对：在 LIBERO 上训练/使用同一 CE-WM 或 suite-specific CE-WM，明确说明训练数据来源；如果不同 model action distribution 差异大，需要校准动作尺度。

---

## 15. 推荐最终实验矩阵

### 最小可投稿版本

| Component | Must Have |
|---|---|
| Dataset | CALVIN + LIBERO-Spatial/Object/Goal/Long |
| Base Models | FLOWER on CALVIN, OpenVLA-OFT on LIBERO, Diffusion Policy or BC-Transformer |
| Main Metric | CALVIN AvgLen/L1-L5, LIBERO success |
| Robustness | CALVIN OOD + LIBERO perturbation |
| Ablation | trust region, accept/reject, uncertainty gate, calibrated CE-WM |
| Diagnostics | intervention rate, action deviation, energy decrease, latency |
| Recovery | at least one failure-state or long-horizon recovery experiment |

### 强顶会版本

| Component | Strong Version |
|---|---|
| Dataset | CALVIN + LIBERO four suites + LIBERO-90 |
| Base Models | FLOWER, OpenVLA-OFT, π0/π0.5, Diffusion Policy, BC-Transformer, optional Octo |
| Robustness | LIBERO-X / LIBERO-Para style perturbations + CALVIN OOD severity |
| Test-time Baselines | action voting, random rerank, uncertainty abstention, adaptive compute baseline |
| Recovery | failure-state recovery + LIBERO-Long partial progress |
| Efficiency | accuracy-latency Pareto |
| Qualitative | rollout videos and failure visualizations |

---

## 16. 文献与仓库清单

### CALVIN / CALVIN baselines

- CALVIN benchmark: https://calvin.cs.uni-freiburg.de/
- CALVIN GitHub: https://github.com/mees/calvin
- HULC: https://github.com/lukashermann/hulc
- HULC paper: https://arxiv.org/abs/2204.06252
- SPIL project: https://hk-zh.github.io/spil/
- SPIL GitHub: https://github.com/hk-zh/spil
- RoboFlamingo GitHub: https://github.com/RoboFlamingo/RoboFlamingo
- GR-1 project: https://gr1-manipulation.github.io/
- GR-1 GitHub: https://github.com/bytedance/GR-1
- SuSIE project: https://rail-berkeley.github.io/susie/
- SuSIE GitHub: https://github.com/kvablack/susie
- MDT GitHub: https://github.com/intuitive-robots/mdt_policy
- MoDE GitHub: https://github.com/intuitive-robots/MoDE_Diffusion_Policy
- MoDE project: https://mbreuss.github.io/MoDE_Diffusion_Policy/
- FLOWER OpenReview: https://openreview.net/forum?id=JeppaebLRD
- FLOWER arXiv: https://arxiv.org/abs/2509.04996
- FLOWER CALVIN GitHub: https://github.com/intuitive-robots/flower_vla_calvin
- FLOWER Hugging Face collection: https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f
- Seer project: https://nimolty.github.io/Seer/
- Seer GitHub: https://github.com/InternRobotics/Seer
- Robot-VLAs/RoboVLMs: https://github.com/Robot-VLAs/RoboVLMs

### LIBERO / LIBERO baselines

- LIBERO official repo: https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO project site: https://libero-project.github.io
- LIBERO datasets: https://libero-project.github.io/datasets
- LIBERO NeurIPS 2023 paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf
- LIBERO policy architectures: https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/policy_architectures.html
- LeRobot LIBERO docs: https://huggingface.co/docs/lerobot/en/libero

### OpenVLA / OpenVLA-OFT

- OpenVLA GitHub: https://github.com/openvla/openvla
- OpenVLA paper: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT project: https://openvla-oft.github.io/

### π0 / π0.5

- OpenPI repo: https://github.com/Physical-Intelligence/openpi
- OpenPI robotics project listing: https://robotics.growbotics.ai/projects/foundation-models/openpi
- RLinf π0/π0.5 docs: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html

### Octo / RT 系列

- Octo project: https://octo-models.github.io/
- Octo GitHub: https://github.com/octo-models/octo
- RT-1 project: https://robotics-transformer1.github.io/
- RT-2 paper: https://robotics-transformer2.github.io/assets/rt2.pdf

### Diffusion Policy / RoboMimic

- Diffusion Policy paper: https://arxiv.org/abs/2303.04137
- Diffusion Policy project: https://diffusion-policy.cs.columbia.edu/
- Diffusion Policy IJRR: https://journals.sagepub.com/doi/10.1177/02783649241273668
- RoboMimic GitHub: https://github.com/ARISE-Initiative/robomimic
- RoboMimic algorithms: https://robomimic.github.io/docs/v0.4/introduction/implemented_algorithms.html
- RoboMimic Transformer docs: https://robomimic.github.io/docs/tutorials/training_transformers.html
- RoboMimic Diffusion Policy docs: https://robomimic.github.io/docs/tutorials/training_diffusion_policy.html

### Robustness / OOD / test-time adaptation

- LIBERO-X: https://arxiv.org/abs/2602.06556
- LIBERO-Para: https://arxiv.org/abs/2603.28301
- Test-Time Perturbation Learning for VLAs: https://arxiv.org/abs/2604.18107
- EVOLVE-VLA: https://showlab.github.io/EVOLVE-VLA/
- HELM: https://arxiv.org/abs/2604.18791
- RePO-VLA: https://papers.cool/arxiv/2605.09410
- Agentic-VLA: https://papers.cool/arxiv/2605.22896
- STRONG-VLA: https://arxiv.org/abs/2604.10055
- Act, Think or Abstain: https://arxiv.org/abs/2603.05147
- VLA-ATTC: https://kurate.org/paper/a4e43ca9-ef4b-46e6-8718-c30cff1858a8
- RobustVLA: https://openreview.net/forum?id=9g7UXLqD4B
- RESample: https://arxiv.org/abs/2510.17640
- ADPro: https://arxiv.org/abs/2508.06266

---

## 17. 结论

CE-AIS 后续实验不应只围绕“FLOWER clean avg length 是否从 3.98 提到 4.04”展开。这个结果可以作为 pilot，但不足以支撑顶会主张。

真正有说服力的实验路线应该是：

1. 用 CALVIN 保留长程语言操作基准，并严肃处理 FLOWER 官方 SOTA 与本地复现差距；
2. 用 LIBERO 建立第二主数据集，覆盖 Spatial/Object/Goal/Long，优先接入 OpenVLA-OFT；
3. 加入至少一个非 VLA action model，如 Diffusion Policy 或 BC-Transformer，证明 model-agnostic；
4. 若可行，加入 π0/π0.5，提升论文时代性和 baseline 强度；
5. 将 OOD robustness、failure recovery、long-horizon partial progress 作为 CE-AIS 的核心亮点；
6. 用消融实验明确证明 trust region、accept/reject、uncertainty gate、calibrated CE-WM 都是必要组件；
7. 用 latency/diagnostics 展示 CE-AIS 是可解释、可控、可部署优化的测试时方法。

如果按这个路线完成，论文叙事会从“一个在 FLOWER 上有小幅提升的工程模块”升级为“一个面向多种 VLA/action model 的冻结测试时能量验证与恢复框架”。这才更接近顶会论文需要的实验逻辑和方法说服力。
