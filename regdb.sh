#!/bin/bash
# ------------------------------------------------------------------
# [RegDB] UA-POT Full Training Script
# Standard Setting: Stage1=20 epochs, Stage2=120 epochs
# Requires ~10 trials for final paper reporting.
# ------------------------------------------------------------------

cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

# Configuration
MODE=${1:-"ua_pot"}     
TEST_MODE=${2:-"v2i"}    # v2i (Vis->IR) or i2v (IR->Vis)
DEVICE=${3:-"0"}

# Define Trials to run (Default: Just Trial 1)
# To run all, use: TRIALS=(1 2 3 4 5 6 7 8 9 10)
TRIALS=(1) 

for TRIAL_ID in "${TRIALS[@]}"; do

    TRIAL_NAME="full_train_trial${TRIAL_ID}"
    SAVE_PATH="regdb_${TEST_MODE}_${TRIAL_NAME}_${MODE}"

    echo "------------------------------------------------"
    echo "Starting RegDB Training (${MODE}) - Trial ${TRIAL_ID}"
    echo "Test Mode: ${TEST_MODE}"
    echo "Save Path: ./saved_regdb_resnet_POT/${SAVE_PATH}"
    echo "------------------------------------------------"

    ARGS="
        --dataset regdb \
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
        --batch-pidnum 5 \
        --pid-numsample 4 \
        --test-batch 128 \
        --num-workers 8 \
        --relabel 1 \
        \
        --lr 0.00045 \
        --weight-decay 0.0005 \
        --milestones 50 70 \
        \
        --tri-weight 0.25 \
        --weak-weight 0.25 \
        --sigma 0.8 \
        --temperature 3.0 \
        \
        --test-mode ${TEST_MODE} \
        --trial ${TRIAL_ID} \
        --gall-mode single \
        --seed 1
    "

    # UA-POT Specifics
    if [ "$MODE" == "ua_pot" ]; then
        ARGS="$ARGS --ot-alpha 0.05 --ot-reg 0.05 --ot-mass 0.8"
    fi

    python3 main.py $ARGS

done