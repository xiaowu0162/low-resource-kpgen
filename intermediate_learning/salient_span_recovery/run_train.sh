#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;
DATA_DIR=${HOME_DIR}/data

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
BART_PATH=${HOME_DIR}/models/bart.base/model.pt

SAVE_DIR=$(date +'%Y%m%d-%H%M')_${DATASET}_checkpoints
mkdir -p $SAVE_DIR/code_backup
cp *.sh ${SAVE_DIR}/code_backup

ALLSHARDS=""

# note that the fairseq version from xiaowu0162 must be used to enable the --ssp_recovery flag

function train-sharding () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=1000
LR=6e-05
MAX_TOKENS=2048
# please adjust this number accordingly if you use more than 1 GPU
UPDATE_FREQ=16
# or comment out "--batch-size" below and add "--max-tokens $MAX_TOKENS" to use dynamic bsz controlled by MAX_TOKENS
PER_DEVICE_BSZ=4
ARCH=bart_base # bart_large
#BART_PATH=bart.base/model.pt # bart.large/model.pt
SAVE_DIR=/local/diwu/kpgen_bart_experiments/${DATASET}_checkpoints

mkdir -p $SAVE_DIR

fairseq-train ${ALLSHARDS} \
--restore-file $BART_PATH \
--batch-size $PER_DEVICE_BSZ \
--task translation \
--truncate-source \
--max-source-positions 1024 --max-target-positions 1024 \
--source-lang source --target-lang target \
--layernorm-embedding \
--share-all-embeddings \
--share-decoder-input-output-embed \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--arch $ARCH \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr-scheduler polynomial_decay --lr $LR \
--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--update-freq $UPDATE_FREQ \
--skip-invalid-size-inputs-valid-test \
--find-unused-parameters --ddp-backend=no_c10d \
--ssp-recovery \
--save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/output.log;

}


while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID DATASET_NAME"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "DATASET_NAME   Name of the training dataset. e.g., kp20k, kptimes, etc."
        echo
        exit;;
   esac
done


N_SHARDS=35

if [[ $2 == 'kp20k-ssr' ]]; then
    ALLSHARDS=$DATA_DIR/scikp/kp20k-salient-span/fairseq/gpt2_bpe_ssr/binary/shard1
    for i in `seq 2 ${N_SHARDS}`; do
	    ALLSHARDS+=":$DATA_DIR/scikp/kp20k-salient-span/fairseq/gpt2_bpe_ssr/binary/shard${i}"
    done
    echo ${ALLSHARDS}
    train-sharding "$1" $2
elif [[ $2 == 'kp20k-ssp' ]]; then
    ALLSHARDS=$DATA_DIR/scikp/kp20k-salient-span/fairseq/gpt2_bpe_ssp/binary/shard1
    for i in `seq 2 ${N_SHARDS}`; do
	    ALLSHARDS+=":$DATA_DIR/scikp/kp20k-salient-span/fairseq/gpt2_bpe_ssp/binary/shard${i}"
    done
    echo ${ALLSHARDS}
    train-sharding "$1" $2
fi
