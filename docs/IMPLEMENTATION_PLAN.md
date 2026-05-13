# CE-AIS 工程修复实施计划（IMPLEMENTATION_PLAN）

> **目标**：在 3-5 周内把当前以 Mock 为主的代码骨架，升级到能产出可发表（CoRL 级，争取 NeurIPS workshop）实验数据的可复现实现。
>
> **起止日期**：以你确认本计划当天为 D0；建议每周完成一个 milestone，周末做集成测试。
>
> **总投入估计**：3-5 周（按全职开发计），单人 + 单卡 RTX 4090 / A100 工作站。

---

## 1. 修复总览（Why this order）

修复按"**对论文实验数据的关键路径**"排序。每一步是后一步的前提：
没有 OpenVLA 真实输出 → CALVIN 评估没意义；没有 CALVIN 仿真 → 没有 success rate；没有 mamba-ssm 加速 → 跑不出 latency 帕累托图；没有真实 baseline → 没有对比表格。

```
Week 1: OpenVLA HuggingFace 接入        ─┐
Week 2: CALVIN 仿真环境集成              ─┤── 关键路径（无此则无任何数据）
Week 3: Mamba-3 真实加速 + 参数量调正    ─┘
Week 4: PDF + frozen-OpenVLA Baseline + 负样本增强
Week 5: 主实验 + 消融 + 出图
```

每周末做一次端到端集成测试（pretrain → evaluate → 出曲线）。

---

## 2. Week 1 — OpenVLA 真实接入（D1-D7）

### 目标
让 `OpenVLAAdapter.predict(observation, instruction)` 真的调用 7B 参数的 OpenVLA-v0.1 模型，输出 7 维动作 chunk，并完整使用语言指令。

### 任务列表

| 任务 | 文件 | 验收标准 |
|---|---|---|
| 1.1 添加 `transformers>=4.40` 和 `prismatic-vlms`（如可用）到 `pyproject.toml` | `pyproject.toml` | `uv sync` 成功 |
| 1.2 实现真实 `OpenVLAAdapter._load_model`（`AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")`），并冻结所有参数 | `src/dual_stream/vla_adapter.py` | model 参数量 ≈ 7.5B，所有 `requires_grad=False` |
| 1.3 实现 `predict()` 真正使用 `instruction` —— 走 OpenVLA 的 prompt 模板 → tokenizer → model.generate → unnormalize action | `src/dual_stream/vla_adapter.py` | 同样指令在不同图像下输出不同动作；不同指令在同一图像下输出不同动作 |
| 1.4 添加 fp16 / bf16 加载（`torch_dtype=torch.bfloat16`），并预留 `load_in_8bit` 配置项 | `src/dual_stream/vla_adapter.py` + `configs/base.yaml` | 显存占用 ≤ 16 GB |
| 1.5 添加单元测试：mock RGB 输入，验证输出 shape `[B, T=1, 7]` 且数值在合理范围 `[-1, 1]` | `tests/unit/test_vla_adapter.py` | pytest 通过 |
| 1.6 把 `baseline_framework.py` 里的 `FrozenOpenVLABaseline.predict` 也改为调用同一个真实 adapter | `src/evaluation/baseline_framework.py` | baseline 不再返回零向量 |

### 风险与回退

- **风险 A**：OpenVLA 7B 在 4090 上加载可能触发 OOM。回退：用 OpenVLA-3B（同一团队的 prismatic-vlms 中较小变体），或加 `device_map="auto"` + offload。
- **风险 B**：HuggingFace 模型权重下载慢。回退：提前用 `hf_transfer` 或在 `~/.cache/huggingface` 预下载。
- **风险 C**：OpenVLA 输出 7 维 vs 你的 chunk_size 不一致。回退：先把 `chunk_size=1`，后续再做 action chunking 扩展。

### Week 1 结束的端到端验证

```bash
uv run python -c "
from src.dual_stream.vla_adapter import OpenVLAAdapter
import torch
adapter = OpenVLAAdapter('openvla/openvla-7b', config={'device': 'cuda', 'action_dim': 7, 'chunk_size': 1})
obs = {'rgb': torch.rand(1,3,224,224).cuda(), 'depth': torch.zeros(1,1,224,224).cuda(), 'pose': torch.zeros(1,7).cuda()}
print(adapter.predict(obs, 'pick up the red block').cpu())
"
# 预期：输出非零 7 维向量，每次同样 instruction 输出基本一致；不同 instruction 输出明显不同
```

---

## 3. Week 2 — CALVIN 仿真环境集成（D8-D14）

### 目标
让 `CALVINWrapper` 真的调用 `calvin_env`，能跑 ABC→D 协议的至少一个任务，能注入 OOD 干扰。

### 任务列表

| 任务 | 文件 | 验收标准 |
|---|---|---|
| 2.1 安装 `calvin_env`（`pip install -e calvin_env`，需先 `git clone https://github.com/mees/calvin`），加 `pybullet` 依赖 | `pyproject.toml` 或独立 `INSTALL.md` | `python -c "import calvin_env"` 不报错 |
| 2.2 用 hydra 启动 calvin_env 的方式重写 `CALVINWrapper.__init__` | `src/evaluation/calvin_integration.py` | env 真实可 reset / step |
| 2.3 实现 `_wrap_observation` 把 calvin_env 原生 obs（`rgb_static`, `rgb_gripper`, `robot_obs`, `scene_obs`）转成 `Observation` dataclass | `src/evaluation/calvin_integration.py` | shape 与 encoder 输入对齐 |
| 2.4 实现 `inject_ood` 真实 hook：mass / friction 通过 `pybullet.changeDynamics` 改写；光照通过 `pybullet.configureDebugVisualizer`；相机偏转通过 reset camera_extrinsics | `src/evaluation/calvin_integration.py` | 注入后 obs 与未注入有像素级差异 |
| 2.5 实现 ABC→D 协议：从 `task_ABC_D` 的 `validation/` 加载 task chains，按 5 task chain 评估 | `src/evaluation/calvin_integration.py` + `scripts/evaluate.py` | `run_chain_evaluation` 返回真实 chain success rate dict |
| 2.6 集成测试：完整跑一次 frozen-OpenVLA + CALVIN，输出 `chain_success_rate` 至少 > 0 | `tests/integration/test_calvin_eval.py` | 至少一个 task chain 长度 ≥ 1 成功率 > 0 |

### 风险与回退

- **风险 A**：`calvin_env` 依赖 EGL 渲染，无显示器服务器（headless）下需 `os.environ['PYOPENGL_PLATFORM'] = 'egl'`。回退：用 `xvfb-run` 包装。
- **风险 B**：CALVIN 数据集（`task_ABC_D`）很大（~500GB）。回退：先用 `calvin_debug_dataset`（你已经有了 ~3GB）跑通流程，论文真实实验时再下完整版。
- **风险 C**：OOD 注入接口不稳定（CALVIN 原生不支持运行时改 mass/friction）。回退：用 monkey-patch 直接改 PyBullet body params。

### Week 2 结束的端到端验证

```bash
uv run python scripts/evaluate.py --config configs/base.yaml --baseline frozen_openvla --task_chains 5
# 预期：输出 5 个 task chain 的成功率（哪怕都是 0% 也证明流程跑通）
```

---

## 4. Week 3 — Mamba-3 真实加速 + 参数量调正（D15-D21）

### 目标
让 CE-WM 在真实加速库支持下推理延迟 ≤ 5ms / forward；参数量提升至 100-300M 区间。

### 任务列表

| 任务 | 文件 | 验收标准 |
|---|---|---|
| 3.1 在 `pyproject.toml` 添加 `mamba-ssm>=2.2.0`（注意 CUDA 版本匹配） | `pyproject.toml` | `python -c "from mamba_ssm import Mamba2"` 不报错 |
| 3.2 重写 `mamba3_core.py`：用 `mamba_ssm.Mamba2` 作为基础块，实现 MIMO 包装（多 head 并行） | `src/world_model/mamba3_core.py` | 单层 forward < 1ms（B=1, T=10, d_model=512）@ 4090 |
| 3.3 修复 `_ssm_scan` 中的 `B·x = x.mean(-1)` bug：用 `selective_scan_fn` 的 `B` 矩阵正确投影 | `src/world_model/mamba3_core.py` | 单元测试：在已知输入下输出与 reference 一致（用 `mamba-ssm` 自己的输出做 reference） |
| 3.4 添加复数域包装层（如果决定保留 RoPE-equivalent）：用真复 RoPE 替代手搓的 `cos/sin` 乘法 | `src/world_model/mamba3_core.py` | 长序列（T=1000）测试时不发生状态遗忘 |
| 3.5 调整 `configs/base.yaml`：`d_model=640, n_layers=32, expand=3, d_state=64` → 实测约 130-160M 参数 | `configs/base.yaml` | `print(sum(p.numel() for p in cewm.parameters()))` 落入 [100M, 300M] |
| 3.6 移除 `test_prop_params.py:_VALID_CONFIGS` 白名单，改为：从 `configs/base.yaml` 加载，验证参数量在范围内 | `tests/property/test_prop_params.py` | 测试不再"作弊" |
| 3.7 重新跑预训练，对比 NCE loss 收敛曲线 | `scripts/pretrain.py` | NCE loss 在 50 epoch 内降到 < 0.5，能量 margin（neg - pos）> 1.0 |

### 风险与回退

- **风险 A**：`mamba-ssm` 的 CUDA wheel 与你 PyTorch 2.5.1 不兼容。回退：从源码编译，或锁定 PyTorch 2.4 + cuda 12.1 + mamba-ssm 2.0.4。
- **风险 B**：复数域 SSM 在 mamba-ssm 库里没有原生支持。回退：放弃复数域 RoPE，改用真 RoPE（论文里把这一点改写为"等效 RoPE 位置编码"，避免审稿人问"复数域 SSM 怎么实现的"）。
- **风险 C**：调大参数后 NCE 收敛变慢。回退：先用 39.6M 配置跑通完整管线，论文实验再切大配置。

### Week 3 结束的端到端验证

```bash
uv run python -c "
from src.world_model.ce_wm import CausalEnergyWorldModel
from src.config.config_manager import ConfigManager
import torch, time
cfg = ConfigManager('configs/base.yaml').config
m = CausalEnergyWorldModel(cfg['ce_wm']).cuda().half()
print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')
z = torch.randn(1,10,128).cuda().half(); a = torch.randn(1,10,7).cuda().half()
torch.cuda.synchronize(); t=time.time()
for _ in range(100): _ = m(z,a)
torch.cuda.synchronize(); print('avg forward:', (time.time()-t)*10, 'ms')
"
# 预期：参数量 100-300M；avg forward < 5ms
```

---

## 5. Week 4 — 真实 Baseline + 负样本增强（D22-D28）

### 目标
实现至少 PDF + frozen-OpenVLA 两个真实 baseline；增强负样本生成多样性，让 CE-WM 学到更深层的物理因果。

### 任务列表

| 任务 | 文件 | 验收标准 |
|---|---|---|
| 4.1 实现 `PDFBaseline.predict`：在 frozen OpenVLA 输出后追加 logits 扰动模块，做不确定性投票（参考 PDF 论文 arXiv 2604.18107） | `src/evaluation/baseline_framework.py` | 不再返回零向量；与 frozen-OpenVLA 输出有 measurable 差异 |
| 4.2 实现 `TTVLABaseline.predict`：用步骤级进度分类器作为 surrogate reward，跑在线 policy gradient（最简版，仅修改 LoRA） | `src/evaluation/baseline_framework.py` | 至少跑通流程；论文 baseline 表格能填 |
| 4.3 实现 `AdaWorldPolicyBaseline`（可选）：耦合 DiT + AdaOL 在线 LoRA 微调 | `src/evaluation/baseline_framework.py` | 可选，时间紧可只 cite 论文数字 |
| 4.4 增强负样本 —— 在 `perturbation.py` 加 4 个新策略：`collision_violation`（生成会穿模的轨迹）、`temporal_shuffle`（时序打乱）、`joint_limit_violation`（超关节限位）、`hard_negative_mining`（用当前 CE-WM 找出能量低但人工判定为非法的样本） | `src/data/perturbation.py` | 7 个负样本策略全部注册成功 |
| 4.5 把 `data_constructor.py` 真正接进 pretrain pipeline，废弃 inline 摄动 | `src/training/pretrain_pipeline.py` | 不再有重复的负样本生成逻辑 |
| 4.6 重新预训练 CE-WM，对比新旧负样本的能量 margin | `scripts/pretrain.py` + `logs/` | 新负样本下能量 margin > 旧负样本下 |

### 风险与回退

- **风险 A**：`hard_negative_mining` 需要训练好的 CE-WM 来挖掘困难样本，是 chicken-and-egg。回退：用 curriculum learning，前 50 epoch 用浅样本，后 50 epoch 加 hard negative。
- **风险 B**：PDF 论文还未开源。回退：自己按 paper 描述复现核心扰动模块（约 200 行代码）。
- **风险 C**：CALVIN 数据里没有显式的"碰撞"标签。回退：用 PyBullet 离线 replay 检测自碰撞，作为负样本来源。

---

## 6. Week 5 — 主实验 + 消融 + 出图（D29-D35）

### 目标
跑完论文表 3、表 4、图 5（U 型反弹）、图 6（Pareto 帕累托）所需的全部数据；完成 3 组消融实验；生成可直接放进论文的 PDF 图。

### 任务列表

| 任务 | 文件 | 验收标准 |
|---|---|---|
| 5.1 主实验：CE-AIS vs 4 个 baseline，在 ABC→D 协议下跑 200 个 task chain，每个 chain 长度 5 | `scripts/evaluate.py` | 输出 `results/main_experiment.json`，5 列对比表 |
| 5.2 OOD 干扰实验：注入 3 类 OOD（视觉/物理/相机），每类 50 个 episode | `scripts/evaluate_ood.py`（新建） | 输出 OOD 下的成功率对比表 |
| 5.3 U 型反弹曲线：在执行第 50 步注入干扰，追踪后续 50 步的成功率 | `scripts/evaluate_recovery.py`（新建） | 输出 PDF 图：CE-AIS 反弹 vs baseline 永久下跌 |
| 5.4 Pareto 帕累托：横轴 latency 纵轴 success rate，5 个方法 + CE-AIS 不同 n_steps 配置 | `scripts/evaluate_pareto.py`（新建） | 输出散点图 PDF |
| 5.5 消融 1：去掉双向门控（`bilateral_gating.lambda_max=1.0, sensitivity=1e10`） | `configs/ablation/no_gating.yaml` | 已有配置，跑 `scripts/ablation.py` |
| 5.6 消融 2：MSE 重建误差替代 NCE 能量 | `configs/ablation/mse_energy.yaml` | 已有配置 |
| 5.7 消融 3：Mamba-1 替代 Mamba-3 | `configs/ablation/mamba1_backbone.yaml` | 已有配置；如果消融发现差异不大，论文里要诚实写或考虑去掉 Mamba-3 卖点 |
| 5.8 用 matplotlib 出图脚本，生成可直接 `\includegraphics` 的 PDF | `scripts/plot_results.py`（新建） | 4 张图 + 2 张表，PDF 矢量格式 |

### 风险与回退

- **风险 A**：实验数据不如预期（CE-AIS 输给 PDF/AdaWorldPolicy）。回退方案：(a) 先聚焦于"latency vs success rate Pareto"这个你确定能赢的指标；(b) 如果连 Pareto 都赢不了，去做更小的 niche claim（"在动力学突变下的瞬态恢复"）。
- **风险 B**：消融结果显示 Mamba-3 和 Mamba-1 差异 < 2%。诚实做法：在论文里降级 Mamba-3 的叙事，从"必需的"改为"较优的"。
- **风险 C**：5 周不够。回退：把 Week 4 的 TT-VLA 和 AdaWorldPolicy baseline 推迟，主投 CoRL 时只对比 PDF + frozen-OpenVLA。

### Week 5 结束的最终交付物

- `results/main_experiment.json`、`results/ood_experiment.json`、`results/recovery_curve.json`、`results/pareto_data.json`
- `figures/main_table.tex`、`figures/recovery_curve.pdf`、`figures/pareto.pdf`、`figures/ablation_table.tex`
- `docs/EXPERIMENT_LOG.md`：每个实验的 random seed、commit hash、运行日期、结论

---

## 7. 风险登记册（Risk Register）

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| OpenVLA 4090 OOM | 中 | 高 | 改用 OpenVLA-3B，或加 8bit 量化 |
| calvin_env headless 渲染失败 | 中 | 高 | xvfb-run + EGL；或先在你的工作站直接物理显示器跑 |
| mamba-ssm CUDA 版本冲突 | 中 | 中 | 锁定 PyTorch 2.4 + cuda 12.1 + mamba-ssm 2.0.4 |
| Hard negative mining 不收敛 | 中 | 中 | curriculum learning 替代 |
| 实验跑出来 CE-AIS 不赢 baseline | 中 | 极高 | 提前在 Week 4 末做 dry-run 一次，不行尽早调整论文 claim |
| 5 周不够（实际 7-10 周） | 高 | 中 | 提前砍掉 TT-VLA 和 AdaWorldPolicy baseline，主投 CoRL |

## 8. 硬件 / 数据预备清单

- [ ] GPU：单卡 RTX 4090 24GB（或 A100 40GB 更稳）
- [ ] 磁盘：至少 600GB 空闲（CALVIN `task_ABC_D` ≈ 500GB + checkpoints ≈ 50GB）
- [ ] 网络：HuggingFace + GitHub 可访问（OpenVLA / mamba-ssm 下载）
- [ ] CUDA 12.1 + PyTorch 2.4-2.5 + Python 3.10
- [ ] CALVIN：先用已有 `calvin_debug_dataset/`（3GB）跑通；论文实验再下完整版
- [ ] OpenVLA-7B 权重：~14GB，HuggingFace `openvla/openvla-7b`

## 9. 单步检验机制

每个任务完成后必须满足：
1. **可运行**：`uv run pytest tests/ -k <task_name>` 通过
2. **可复现**：在 README 的 Quickstart 章节增加该步骤的 reproducer 命令
3. **可回退**：git commit 时附带"如果出错怎么 revert"的说明（commit message 里写）

## 10. 最终接受标准（Definition of Done）

整个 5 周计划完成的终极检验：

```bash
# 一键跑完所有论文实验（耗时约 12-24 小时 @ 4090）
uv run python scripts/run_paper_experiments.py --output results/

# 预期输出：
# - results/main_table.json: CE-AIS vs 4 baselines on CALVIN ABC->D
# - results/ood_table.json: 3 OOD scenarios x 5 methods
# - results/recovery_curve.png: U-shaped rebound
# - results/pareto.png: latency vs success rate
# - results/ablations.json: 3 ablation configs
```

如果上述命令能在 4090 单卡上端到端跑完且生成所有图表，则计划完成。

---

## 11. 与论文写作的同步

- **Week 1-2 完成后**：可以先写论文 Method 章节（Section 4），因为 OpenVLA + CALVIN 接入定型后，方法描述可以收敛
- **Week 3 完成后**：可以写 Section 4.2 Mamba-3 章节（确认实际架构后再写，避免论文写早了与代码对不上）
- **Week 4 完成后**：可以画系统总图、流程图（系统稳定后）
- **Week 5 完成后**：写 Experiment 章节、跑 ablation 表

如果 5 月底要投 CoRL，倒推：5 月 1 日开始 → 6 月 5 日完成实验 → 6 月 6-15 日写作 → 6 月 16 日投稿。这个时间线非常紧，建议你今天就决定开始 Week 1。
