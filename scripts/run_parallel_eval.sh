#!/bin/bash
# 并行评估脚本：单卡多进程跑 CALVIN 官方 1000 chains
# 用法: bash scripts/run_parallel_eval.sh [GPU_ID] [N_WORKERS] [METHODS] [VLA_TYPE]
# 示例:
#   bash scripts/run_parallel_eval.sh 7 4 "frozen_flower ce_ais" flower
#   bash scripts/run_parallel_eval.sh 6 4 "frozen_robovlms" robovlms

set -e

GPU_ID=${1:-7}
N_WORKERS=${2:-4}
METHODS=${3:-"frozen_flower ce_ais"}
VLA_TYPE=${4:-"flower"}
TOTAL_CHAINS=1000
CHAINS_PER_WORKER=$((TOTAL_CHAINS / N_WORKERS))
OUTPUT_BASE="results/${VLA_TYPE}_official_1000chains"

cd /data0/yejinxuan/ce-ais
mkdir -p "$OUTPUT_BASE"

export HF_HOME=/data0/yejinxuan/hf_cache
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data0/yejinxuan/ce-ais/external/RoboVLMs:/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH
export DISPLAY=""

# VLA 类型对应的参数
VLA_ARGS=""
if [ "$VLA_TYPE" = "flower" ]; then
    VLA_ARGS="--flower-checkpoint-dir data/flower_calvin_abc --flower-code-path external/flower_vla_calvin"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
elif [ "$VLA_TYPE" = "robovlms" ]; then
    VLA_ARGS="--robovlms-checkpoint-dir data/robovlms --robovlms-code-path external/RoboVLMs"
fi

echo "======================================"
echo "Parallel CALVIN Evaluation"
echo "GPU: $GPU_ID | Workers: $N_WORKERS | Chains/worker: $CHAINS_PER_WORKER"
echo "VLA: $VLA_TYPE | Methods: $METHODS"
echo "Output: $OUTPUT_BASE"
echo "======================================"

PIDS=()

for i in $(seq 0 $((N_WORKERS - 1))); do
    OFFSET=$((i * CHAINS_PER_WORKER))
    WORKER_DIR="${OUTPUT_BASE}/worker_${i}"
    LOG_FILE="${OUTPUT_BASE}/worker_${i}.log"

    echo "[Worker $i] chains ${OFFSET}...$((OFFSET + CHAINS_PER_WORKER - 1)) -> $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python -u scripts/run_paper_experiments.py \
        --data-dir data/task_ABC_D \
        --vla-type $VLA_TYPE \
        $VLA_ARGS \
        --methods $METHODS \
        --sequence-source flower_official \
        --n-chains $CHAINS_PER_WORKER \
        --chain-offset $OFFSET \
        --chain-length 5 \
        --max-steps 360 \
        --config configs/safe_balanced.yaml \
        --device cuda:$GPU_ID \
        --encoder-ckpt checkpoints/encoder_epoch0044.pt \
        --cewm-ckpt checkpoints_calibrated_cewm/cewm_epoch0033.pt \
        --output-dir "$WORKER_DIR" \
        --seed 42 \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $N_WORKERS workers launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
echo ""

# 等待所有进程，实时报告完成状态
FAILED=0
for i in $(seq 0 $((N_WORKERS - 1))); do
    if wait ${PIDS[$i]}; then
        echo "[Worker $i] DONE (PID ${PIDS[$i]})"
    else
        echo "[Worker $i] FAILED (PID ${PIDS[$i]})"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED worker(s) failed. Check logs in $OUTPUT_BASE/worker_*.log"
    exit 1
fi

echo ""
echo "All workers completed. Merging results..."
echo ""

# 合并结果
.venv/bin/python -u scripts/merge_parallel_results.py \
    --output-base "$OUTPUT_BASE" \
    --n-workers $N_WORKERS

echo ""
echo "Done! Final results in: ${OUTPUT_BASE}/merged_results.json"
