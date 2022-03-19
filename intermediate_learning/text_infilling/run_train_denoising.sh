#!/usr/bin/env bash

SRCDIR=data
mkdir -p logs

function train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
TOTAL_NUM_UPDATES=2000000
WARMUP_UPDATES=1000
LR=3e-05
MAX_TOKENS=2048
BATCH_SIZE=64
UPDATE_FREQ=8
ARCH=bart_base # bart_large
BART_PATH=bart.base/model.pt              #bart.base/model.pt   bart.large/model.pt
SAVE_DIR=/local/diwu/kpgen_bart_experiments/${DATASET}_denoising_word_span_checkpoints

mkdir -p $SAVE_DIR

fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--restore-file $BART_PATH \
--arch $ARCH \
--max-tokens $MAX_TOKENS \
--memory-efficient-fp16 \
--train-subset train \
--valid-subset valid \
--batch-size $BATCH_SIZE \
--layernorm-embedding \
--share-all-embeddings \
--share-decoder-input-output-embed \
--bpe gpt2 \
--task denoising \
--insert 0 \
--permute-sentences 0 \
--poisson-lambda 3.5 \
--mask 0.3 \
--mask-length 'span-poisson' \
--replace-length 1 \
--rotate 0 \
--mask-random 0.1 \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--criterion cross_entropy \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr-scheduler polynomial_decay --lr $LR \
--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--update-freq $UPDATE_FREQ \
--skip-invalid-size-inputs-valid-test \
--find-unused-parameters --ddp-backend=no_c10d \
--save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/output.log;

}

function decode () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR_PREFIX=$3

python decode.py \
--data_name_or_path "$SRCDIR/${DATASET}-bin/" \
--data_dir "$SRCDIR/${DATASET}/" \
--checkpoint_dir ${SAVE_DIR_PREFIX}_checkpoints \
--checkpoint_file checkpoint_best.pt \
--output_file logs/${DATASET}_hypotheses.txt \
--batch_size 64 \
--beam 1 \
--min_len 16 \
--lenpen 1.0 \
--no_repeat_ngram_size 3 \
--max_len_b 60;

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


if [[ $2 == 'kptimes' ]]; then
    train "$1" $2
    # decode "$1" $2 $2
elif [[ $2 == 'stepcontent-mixed-kptimes' ]]; then
    train "$1" $2
elif [[ $2 == 'taboola-stepcontent' ]]; then
    train "$1" $2
elif [[ $2 == 'kp-jptimes-full' ]]; then
    train "$1" $2  
elif [[ $2 == 'jptimes-5000' ]]; then
    train "$1" $2  
elif [[ $2 == 'kp20k' ]]; then
    train "$1" $2  
elif [[ $2 == 'kp-jptimes-full-balanced' ]]; then
    train "$1" $2  
else
    echo "Unrecognized dataset."
fi
