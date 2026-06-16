#!/bin/bash
# RoboFlamingo CALVIN ABC→D 评估脚本（单GPU）
# Usage: CUDA_VISIBLE_DEVICES=7 bash scripts/eval_roboflamingo.sh

set -e
cd /data0/yejinxuan/ce-ais

export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=4.1
export PYTHONUNBUFFERED=1
export HF_HOME=/data0/yejinxuan/hf_cache
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset NCCL_BLOCKING_WAIT
export TORCH_NCCL_BLOCKING_WAIT=1

# 从 CUDA_VISIBLE_DEVICES 推断 EGL device
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export EGL_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
fi

# Paths
ROBOFLAMINGO_ROOT=external/RoboFlamingo
CALVIN_DATASET=/data0/yejinxuan/ce-ais/data/task_ABC_D
CALVIN_CONF=/data0/yejinxuan/workspace/calvin/calvin_models/conf
CHECKPOINT=/data0/yejinxuan/ce-ais/data/RoboFlamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth
LOG_FILE=results/roboflamingo_eval.log

export PYTHONPATH=${ROBOFLAMINGO_ROOT}:${ROBOFLAMINGO_ROOT}/open_flamingo:${ROBOFLAMINGO_ROOT}/robot_flamingo/eval:external/flower_vla_calvin/calvin_env:/data0/yejinxuan/workspace/calvin/calvin_models:$PYTHONPATH

mkdir -p results

echo "======================================"
echo "RoboFlamingo CALVIN ABC→D Evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $CALVIN_DATASET"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "EGL_VISIBLE_DEVICES: $EGL_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"
echo "======================================"

# 单GPU评估（nproc_per_node=1）
.venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=29501 \
    ${ROBOFLAMINGO_ROOT}/robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RoboFlamingo_ABC_D \
    --calvin_dataset ${CALVIN_DATASET} \
    --calvin_conf_path ${CALVIN_CONF} \
    --cross_attn_every_n_layers 1 \
    --evaluate_from_checkpoint ${CHECKPOINT} \
    --workers 1 \
    > ${LOG_FILE} 2>&1

echo "Done. Results in ${LOG_FILE}"
