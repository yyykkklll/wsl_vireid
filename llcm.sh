#!/bin/bash
# ------------------------------------------------------------------
# [LLCM] UA-POT Full Training Script
# ------------------------------------------------------------------

cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

# Configuration
MODE=${1:-"ua_pot"}
TEST_MODE=${2:-"v2t"}    # v2t (Vis->Thermal) is standard for LLCM
DEVICE=${3:-"0"}

TRIAL_NAME="full_train_$(date +%m%d_%H%M)"
SAVE_PATH="llcm_${TEST_MODE}_${TRIAL_NAME}_${MODE}"

echo "------------------------------------------------"
echo "Starting LLCM Training (${MODE})"
echo "Test Mode: ${TEST_MODE}"
echo "Save Path: ./saved_llcm_resnet_POT/${SAVE_PATH}"
echo "------------------------------------------------"

ARGS="
    --dataset llcm \
    --arch resnet \
    --mode train \
    --data-path ./datasets/ \
    --save-path ${SAVE_PATH} \
    --device ${DEVICE} \
    \
    --stage1-epoch 20 \
    --stage2-epoch 120 \
    \
    --img-h 288 \
    --img-w 144 \
    --batch-pidnum 8 \
    --pid-numsample 4 \
    --test-batch 128 \
    --num-workers 8 \
    --relabel 1 \
    \
    --lr 0.0003 \
    --weight-decay 0.0005 \
    --milestones 30 70 \
    \
    --tri-weight 0.25 \
    --weak-weight 0.25 \
    --sigma 0.8 \
    --temperature 3.0 \
    \
    --test-mode ${TEST_MODE} \
    --gall-mode single \
    --seed 1
"

# UA-POT Specifics
if [ "$MODE" == "ua_pot" ]; then
    ARGS="$ARGS --ot-alpha 0.05 --ot-reg 0.05 --ot-mass 0.8"
fi

python3 main.py $ARGS