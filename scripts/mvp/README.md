# CE-AIS MVP（Minimum Viable Proof）

> **目的**：在投入 7 周大计划前，用 5-7 天**用最小成本**验证 CE-AIS 的核心假设：
>
> > **NCE 训出的小 CE-WM 能在 OOD 视觉干扰下，通过 EFE 朗之万偏转，把"被污染的脆弱 VLA"输出拉回到接近专家轨迹的方向，且改善幅度 ≥ 2-3%。**
>
> 如果 MVP 通过 → 启动 7 周大计划，信心十足；
> 如果 MVP 失败 → 立即调整方法（换损失 / 换负样本 / 重设计），不浪费 6 周。

## 设计哲学

MVP 故意 **降低工程复杂度**，与最终论文实验**完全不同**：

| 维度 | 最终论文 | MVP |
|---|---|---|
| VLA 基座 | OpenVLA-7B | 1M 参数 BC proxy（自己训的小 MLP，模拟"会被 OOD 击垮的 VLA"）|
| CE-WM 规模 | 100-300M | 5M（n_layers=4, d_model=128）|
| 仿真环境 | 真实 calvin_env | **离线 replay 评估**（用 calvin_debug 数据集） |
| 评估指标 | task chain success rate | **action MSE 相对改善率**（连续值代理） |
| OOD 注入 | 物理动力学 + 视觉灾难 | 仅视觉（亮度 shift + 高斯噪声）|
| 训练时长 | 数天 | < 1 小时 |
| 总耗时 | 7 周 | 5-7 天 |

## 成功标准

> **PRIMARY**：在 OOD 条件下，"steered" 配置的 action MSE 相对 "proxy" 配置降低 **≥ 3%**。
>
> **SECONDARY**：在干净条件下，steered 配置的 action MSE 不比 proxy 配置升高超过 **2%**（即 steering 不破坏 clean 性能）。
>
> **STRETCH**：epistemic value（升级 1）开启时相对关闭时再降 ≥ 1%（验证完整 EFE 的边际贡献）。

如果 PRIMARY 通过：✅ 核心思想 work，启动 7 周大计划。
如果 PRIMARY 失败但 SECONDARY 通过：⚠️ steering 不伤害但也不帮助 → 调试 NCE 训练 / 换负样本策略再试一次。
如果两者都失败：❌ 核心假设有问题 → 暂停，重新设计（可能要换 EBM 损失类型或改用扩散引导）。

## Day 1-7 时间表

| Day | 任务 | 产出 |
|---|---|---|
| **Day 1** | 跑 `00_check_env.py`，确认数据/GPU 都 OK；通读 README | 环境就绪 |
| **Day 2** | 跑 `01_train_bc.py` 训 BC proxy（5-10 min）| `checkpoints/mvp/bc_proxy.pt` |
| **Day 3** | 跑 `02_train_cewm.py` 训小 CE-WM（30-60 min）；观察 NCE loss 收敛 | `checkpoints/mvp/cewm_mvp.pt` + 能量 margin > 1.0 |
| **Day 4** | 跑 `03_eval_mvp.py` 主评估（10 min）；查看 4 条件 MSE 对比 | `results/mvp/mvp_results.json` |
| **Day 5** | 调参：朗之万 step_size / kl_weight / OOD 强度，跑 ablation | 最优配置 |
| **Day 6** | 关 / 开 epistemic value 对比（升级 1 验证）| epistemic 边际贡献数据 |
| **Day 7** | 写一页 MVP 报告，决定 GO / NO-GO | `results/mvp/mvp_report.md` |

## 一键运行

```bash
cd /home/huangyixuan/yejinxuan/ce-ais
bash scripts/mvp/run_mvp.sh
```

或分步运行（推荐第一次手动跑，方便看每步输出）：

```bash
uv run python scripts/mvp/00_check_env.py
uv run python scripts/mvp/01_train_bc.py        # ~5-10 min
uv run python scripts/mvp/02_train_cewm.py      # ~30-60 min
uv run python scripts/mvp/03_eval_mvp.py        # ~10 min
```

## 输出预期

成功的 `03_eval_mvp.py` 输出大致长这样：

```
================== CE-AIS MVP RESULTS ==================
                        clean MSE    OOD MSE    OOD relative gain
proxy_only              0.0142       0.0287     baseline
steered (no epistemic)  0.0145       0.0264     -8.0%  ✅
steered (full EFE)      0.0143       0.0258     -10.1% ✅✅

PRIMARY criterion (OOD gain >= 3%):  PASS
SECONDARY criterion (clean delta <= 2%): PASS  
STRETCH criterion (epistemic gain >= 1%): PASS

→ MVP PASS. Recommend proceeding to Week 1 of full plan.
========================================================
```

## 回退方案

如果某一步失败，按下面顺序排查：

1. **GPU OOM**：脚本里把 `BATCH_SIZE` 从 32 降到 8 或 4
2. **NCE loss 不收敛**（margin < 0.5）：检查 perturbation 是否真的产生了"非法"动作；尝试加大 `noise_std`
3. **steered MSE 不下降**：先确认 CE-WM 在 expert 动作上确实是低能量；检查朗之万 `step_size` 是否合适（建议从 1e-3 试到 1e-1）
4. **Encoder 输出全零**：检查 RGB 范围是否归一化到 [0,1]

## 依赖

- 已有：torch + numpy + (existing src/) + calvin_debug_dataset
- 新装：**无**（这是 MVP 的关键设计——不接 OpenVLA、不接 calvin_env、不装 mamba-ssm）

## MVP 与最终论文的关系

MVP 跑通的代码可以**几乎全部复用**到 7 周大计划中：
- 同一个 ContrastiveEncoder（只是换更大输入）
- 同一个 CausalEnergyWorldModel（只是换更大配置）
- 同一个 langevin_dynamics（只是接到真实 OpenVLA）

MVP **只增加** `01_train_bc.py`（这个在大计划里被 OpenVLAAdapter 替代）和 `03_eval_mvp.py` 的离线 replay 评估（被 calvin_env 真仿真替代）。
