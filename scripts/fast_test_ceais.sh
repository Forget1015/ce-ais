#!/bin/bash
# CE-AIS 快速迭代测试脚本
# 用 28 条代表性 chains 验证参数改动，约 30-40 分钟
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6 bash scripts/fast_test_ceais.sh [config_path] [tag]
#
# Examples:
#   CUDA_VISIBLE_DEVICES=6 bash scripts/fast_test_ceais.sh configs/base.yaml baseline
#   CUDA_VISIBLE_DEVICES=6 bash scripts/fast_test_ceais.sh configs/ceais_high_passthrough.yaml high_pt
#
# 输出: results/fast_test/<tag>/main_experiment.json

set -e
cd /data0/yejinxuan/ce-ais

CONFIG=${1:-configs/base.yaml}
TAG=${2:-$(basename $CONFIG .yaml)}
OUTPUT_DIR=results/fast_test/${TAG}

export PYTHONPATH=external/flower_vla_calvin/calvin_env:external/flower_vla_calvin:$PYTHONPATH
export HF_HOME=/data0/yejinxuan/hf_cache
export PYOPENGL_PLATFORM=egl
export DISPLAY=""
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

mkdir -p ${OUTPUT_DIR}

echo "======================================"
echo "CE-AIS Fast Test"
echo "Config: $CONFIG"
echo "Tag: $TAG"
echo "Output: $OUTPUT_DIR"
echo "Chains: 28 (18 regression + 10 golden)"
echo "======================================"

# 跑 frozen_flower baseline
echo "[1/2] Running frozen_flower baseline..."
.venv/bin/python -u scripts/run_paper_experiments.py \
    --data-dir data/task_ABC_D \
    --vla-type flower \
    --flower-checkpoint-dir data/flower_calvin_abc \
    --flower-code-path external/flower_vla_calvin \
    --methods frozen_flower \
    --sequence-source flower_official \
    --chain-indices configs/fast_test_subsets.json \
    --chain-length 5 --max-steps 360 \
    --device cuda:0 \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR}/frozen \
    --seed 42 \
    > ${OUTPUT_DIR}/frozen.log 2>&1

echo "[1/2] frozen_flower done."

# 跑 ce_ais
echo "[2/2] Running ce_ais..."
.venv/bin/python -u scripts/run_paper_experiments.py \
    --data-dir data/task_ABC_D \
    --vla-type flower \
    --flower-checkpoint-dir data/flower_calvin_abc \
    --flower-code-path external/flower_vla_calvin \
    --methods ce_ais \
    --sequence-source flower_official \
    --chain-indices configs/fast_test_subsets.json \
    --chain-length 5 --max-steps 360 \
    --device cuda:0 \
    --config ${CONFIG} \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --cewm-ckpt checkpoints_calibrated_cewm/cewm_epoch0033.pt \
    --output-dir ${OUTPUT_DIR}/ceais \
    --seed 42 \
    > ${OUTPUT_DIR}/ceais.log 2>&1

echo "[2/2] ce_ais done."

# 分析结果
echo ""
echo "======================================"
echo "Results Analysis"
echo "======================================"
.venv/bin/python -c "
import json, os
import numpy as np

subsets = json.load(open('configs/fast_test_subsets.json'))
regression_indices = set(subsets['regression'])
golden_indices = set(subsets['golden'])
combined = subsets['combined']

def load(path):
    with open(path) as f:
        d = json.load(f)
    for method, res in d['results'].items():
        return res['per_chain_completed_tasks']

frozen = load('${OUTPUT_DIR}/frozen/main_experiment.json')
ceais = load('${OUTPUT_DIR}/ceais/main_experiment.json')

print(f'Total chains: {len(combined)}')
print(f'')
print(f'  {\"Metric\":<20} {\"Frozen\":>8} {\"CE-AIS\":>8} {\"Δ\":>8}')
print(f'  {\"-\"*50}')
avg_f = np.mean(frozen)
avg_c = np.mean(ceais)
print(f'  {\"Avg Len\":<20} {avg_f:>8.3f} {avg_c:>8.3f} {avg_c-avg_f:>+8.3f}')
for l in range(1, 6):
    f_rate = sum(1 for c in frozen if c >= l) / len(frozen) * 100
    c_rate = sum(1 for c in ceais if c >= l) / len(ceais) * 100
    print(f'  {f\"L{l}\":<20} {f_rate:>7.1f}% {c_rate:>7.1f}% {c_rate-f_rate:>+7.1f}%')

# Per-subset analysis
print(f'')
print(f'  Regression subset (18 chains, flower全通过 gated原来失败):')
reg_start = 0
reg_frozen = []
reg_ceais = []
for i, idx in enumerate(combined):
    if idx in regression_indices:
        reg_frozen.append(frozen[i])
        reg_ceais.append(ceais[i])
reg_full = sum(1 for c in reg_ceais if c == 5)
print(f'    CE-AIS 全通过: {reg_full}/{len(reg_ceais)} (目标: 越高越好, 之前全是0)')
print(f'    CE-AIS Avg Len: {np.mean(reg_ceais):.3f} (Frozen: {np.mean(reg_frozen):.3f})')

print(f'')
print(f'  Golden subset (10 chains, flower原来失败 gated修复了):')
gold_frozen = []
gold_ceais = []
for i, idx in enumerate(combined):
    if idx in golden_indices:
        gold_frozen.append(frozen[i])
        gold_ceais.append(ceais[i])
gold_full = sum(1 for c in gold_ceais if c == 5)
print(f'    CE-AIS 全通过: {gold_full}/{len(gold_ceais)} (目标: 保持高, 之前=10)')
print(f'    CE-AIS Avg Len: {np.mean(gold_ceais):.3f} (Frozen: {np.mean(gold_frozen):.3f})')
"

echo ""
echo "Done! Full logs: ${OUTPUT_DIR}/"
