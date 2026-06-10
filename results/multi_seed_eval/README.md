# FLOWER Multi-Seed Evaluation
## 目的
用不同随机种子跑 1000 chains × 5 tasks，统计 avg completed length 的分布。
验证 HF checkpoint 的真实水平。

## 配置
- Checkpoint: data/flower_calvin_abc (HF safetensors, no EMA)
- calvin_env: commit 797142c (with button fix + action copy fix)
- Sequences: flower_official (1000 chains, deterministic)
- GPU: 6 and 7, ~10 processes per GPU
- chain_length: 5, max_steps: 360

## 日志
- seed_XXX.log: 每个 seed 的完整日志
- summary.json: 汇总结果（跑完后生成）
