#!/bin/bash
# ------------------------------------------------------------------
# [SYSU-MM01] UA-POT Full Training Script
# Standard Setting: Stage1=20 epochs, Stage2=120 epochs
# ------------------------------------------------------------------

# Setup
cd "$(dirname "$0")" || exit
export PYTHONPATH=$PYTHONPATH:.

# Configuration
MODE=${1:-"ua_pot"}      # ua_pot or simple_pot
SEARCH_MODE=${2:-"all"}  # all or indoor
DEVICE=${3:-"0"}

# Log Name
TRIAL_NAME="full_train_$(date +%m%d_%H%M)"
SAVE_PATH="sysu_${SEARCH_MODE}_${TRIAL_NAME}_${MODE}"

echo "------------------------------------------------"
echo "Starting SYSU-MM01 Full Training (${MODE})"
echo "Search Mode: ${SEARCH_MODE}"
echo "Save Path:   ./saved_sysu_resnet_POT/${SAVE_PATH}"
echo "Device:      GPU ${DEVICE}"
echo "------------------------------------------------"

# Arguments
ARGS="
    --dataset sysu \
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
    --search-mode ${SEARCH_MODE} \
    --gall-mode single \
    --test-mode t2v \
    --seed 1
"

# UA-POT Specifics
if [ "$MODE" == "ua_pot" ]; then
    ARGS="$ARGS --ot-alpha 0.05 --ot-reg 0.05 --ot-mass 0.8"
elif [ "$MODE" == "simple_pot" ]; then
    ARGS="$ARGS --ot-alpha 0.0 --ot-reg 0.05 --ot-mass 0.8"
fi

# Run
python3 main.py $ARGS