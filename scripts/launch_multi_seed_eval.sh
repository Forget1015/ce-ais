#!/bin/bash
# Multi-seed FLOWER evaluation launcher
# Usage: bash scripts/launch_multi_seed_eval.sh [START_SEED] [NUM_SEEDS] [PROCS_PER_GPU]
#
# Launches parallel evaluations across GPU 6 and GPU 7.
# Each process runs 1000 chains × 5 tasks with a different random seed.

START_SEED=${1:-1}
NUM_SEEDS=${2:-50}
PROCS_PER_GPU=${3:-10}

RESULT_DIR="results/multi_seed_eval"
mkdir -p "$RESULT_DIR"

export HF_HOME=/data0/yejinxuan/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/data0/yejinxuan/ce-ais/external/flower_vla_calvin:/data0/yejinxuan/workspace/calvin/calvin_models:/data0/yejinxuan/workspace/calvin/calvin_env:$PYTHONPATH
export PYTHONUNBUFFERED=1
export DISPLAY=""

TOTAL_PROCS=$((PROCS_PER_GPU * 2))
PIDS=()

echo "$(date): Launching $NUM_SEEDS seeds (start=$START_SEED) with $PROCS_PER_GPU procs/GPU" | tee -a "$RESULT_DIR/launcher.log"

launch_one() {
    local SEED=$1
    local GPU=$2
    local LOG="$RESULT_DIR/seed_${SEED}.log"

    if [ -f "$LOG" ] && grep -q "frozen_flower.*1000/1000" "$LOG" 2>/dev/null; then
        echo "  Seed $SEED already complete, skipping"
        return 1
    fi

    CUDA_VISIBLE_DEVICES=$GPU \
    /data0/yejinxuan/ce-ais/.venv/bin/python -u scripts/run_paper_experiments.py \
        --data-dir data/task_ABC_D \
        --vla-type flower \
        --flower-checkpoint-dir data/flower_calvin_abc \
        --flower-code-path external/flower_vla_calvin \
        --methods frozen_flower \
        --sequence-source flower_official \
        --n-chains 1000 \
        --chain-length 5 \
        --max-steps 360 \
        --device "cuda:$GPU" \
        --encoder-ckpt checkpoints/encoder_epoch0044.pt \
        --cewm-ckpt checkpoints_calibrated_cewm/cewm_epoch0033.pt \
        --output-dir "$RESULT_DIR/seed_${SEED}" \
        --seed $SEED \
        > "$LOG" 2>&1 &

    echo $!
}

# Launch initial batch
SEED_IDX=0
RUNNING=0

for ((i=0; i<TOTAL_PROCS && SEED_IDX<NUM_SEEDS; i++)); do
    SEED=$((START_SEED + SEED_IDX))
    if ((i < PROCS_PER_GPU)); then
        GPU=6
    else
        GPU=7
    fi
    PID=$(launch_one $SEED $GPU)
    if [ -n "$PID" ]; then
        PIDS+=($PID)
        echo "  Launched seed=$SEED on GPU $GPU (PID=$PID)" | tee -a "$RESULT_DIR/launcher.log"
        RUNNING=$((RUNNING + 1))
    fi
    SEED_IDX=$((SEED_IDX + 1))
done

echo "$(date): Initial batch: $RUNNING processes running, next seed index=$SEED_IDX" | tee -a "$RESULT_DIR/launcher.log"

# Wait and refill: when a process finishes, launch the next seed
while ((SEED_IDX < NUM_SEEDS)); do
    # Wait for any child to finish
    wait -n 2>/dev/null

    # Find which PIDs are done
    NEW_PIDS=()
    for PID in "${PIDS[@]}"; do
        if kill -0 $PID 2>/dev/null; then
            NEW_PIDS+=($PID)
        else
            # A slot freed up - figure out which GPU has fewer processes
            GPU6_COUNT=$(echo "${NEW_PIDS[@]}" | tr ' ' '\n' | while read p; do
                cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep "CUDA_VISIBLE_DEVICES=6" && echo 1
            done | wc -l)

            if ((SEED_IDX < NUM_SEEDS)); then
                SEED=$((START_SEED + SEED_IDX))
                # Alternate GPUs
                if (( (SEED_IDX % 2) == 0 )); then
                    GPU=6
                else
                    GPU=7
                fi
                PID=$(launch_one $SEED $GPU)
                if [ -n "$PID" ]; then
                    NEW_PIDS+=($PID)
                    echo "$(date): Launched seed=$SEED on GPU $GPU (PID=$PID)" | tee -a "$RESULT_DIR/launcher.log"
                fi
                SEED_IDX=$((SEED_IDX + 1))
            fi
        fi
    done
    PIDS=("${NEW_PIDS[@]}")
    sleep 5
done

# Wait for all remaining
echo "$(date): All seeds launched, waiting for remaining processes..." | tee -a "$RESULT_DIR/launcher.log"
wait

echo "$(date): All $NUM_SEEDS seeds complete!" | tee -a "$RESULT_DIR/launcher.log"
