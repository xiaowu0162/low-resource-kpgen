#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
BART_PATH=${3:-"${HOME_DIR}/models/bart.base/model.pt"}

SAVE_DIR=$(date +'%Y%m%d-%H%M')_${DATASET}_checkpoints
mkdir -p $SAVE_DIR/code_backup
cp *.sh ${SAVE_DIR}/code_backup


function train () {

TOTAL_NUM_UPDATES=3000   # this is for a 20k-sized dataset; moderate it to fit your need
WARMUP_UPDATES=150
LR=1e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
# or comment out "--batch-size" below and add "--max-tokens $MAX_TOKENS" to use dynamic bsz controlled by MAX_TOKENS
PER_DEVICE_BSZ=8
ARCH=bart_base # bart_large
DATA_DIR=${DATA_DIR_PREFIX}/${DATASET}/fairseq/gpt2_bpe/binary/


fairseq-train $DATA_DIR \
    --restore-file $BART_PATH \
    --task translation \
    --truncate-source \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --source-lang source \
    --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --required-batch-size-multiple 1 \
    --batch-size ${PER_DEVICE_BSZ} \
    --arch $ARCH \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --max-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --save-dir $SAVE_DIR \
    --log-format json \
    2>&1 | tee $SAVE_DIR/finetune.log;

}


while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run_train.sh GPU_ID DATASET_NAME [WARMSTART_CKPT]"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "DATASET_NAME   Name of the training dataset. e.g., kp20k, kptimes, etc."
        echo "WARMSTART_CKPT (Optional) The checkpoint for initializing the model's weights"
        exit;;
   esac
done


# The three low resource kp20k subsets we created are called kp20k-20k-1/2/3
if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp/
    #train
elif [[ $DATASET == 'kp20k-20k-1' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/kp20k-20k/
    train
elif [[ $DATASET == 'kp20k-20k-2' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/kp20k-20k/
    train
elif [[ $DATASET == 'kp20k-20k-3' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/kp20k-20k/
    train
elif [[ $DATASET == 'kptimes' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/
    train
else
    echo "Dataset name not recognized."
fi
