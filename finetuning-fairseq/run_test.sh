#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR=$3


function decode () {

EVAL_DATASET=$1
DATA_DIR=${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq/gpt2_bpe/binary/
OUT_FILE=$SAVE_DIR/${EVAL_DATASET}_out.txt
HYP_FILE=$SAVE_DIR/${EVAL_DATASET}_hypotheses.txt

fairseq-generate $DATA_DIR \
    --path $SAVE_DIR/checkpoint_best.pt \
    --task translation \
    --batch-size 64 \
    --beam 1 \
    --no-repeat-ngram-size 0 \
    --max-len-b 60 \
    2>&1 | tee $OUT_FILE;

grep ^H $OUT_FILE | sort -V | cut -f3- > $HYP_FILE;

}


function text-decode () {

EVAL_DATASET=$1
HYP_FILE=$SAVE_DIR/${EVAL_DATASET}_hypotheses.txt
HYP_FILE_OLD=$SAVE_DIR/${EVAL_DATASET}_hypotheses_raw.txt

mv $HYP_FILE $HYP_FILE_OLD

python -W ignore fairseq_gpt2_decode.py $HYP_FILE_OLD $HYP_FILE ;

}


function evaluate () {

EVAL_DATASET=$1

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_dir ${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq \
    --file_prefix $SAVE_DIR/${EVAL_DATASET} \
    --tgt_dir $SAVE_DIR \
    --log_file $EVAL_DATASET \
    --k_list 5 M;

}


while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run_test.sh GPU_ID DATASET_NAME SAVE_DIR"
        echo
        echo "GPU_ID         An integer (works best with single GPU) "
        echo "DATASET_NAME   Name of the evaluation dataset. e.g., kp20k, kptimes, etc."
        echo "SAVE_DIR       The checkpoint for the trained model"
        exit;;
   esac
done



if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp/
    for dataset in kp20k inspec krapivin nus semeval; do
	decode $dataset
	text-decode $dataset
	evaluate $dataset
    done
elif [[ $DATASET == 'kptimes' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/
    decode $DATASET
    text-decode $dataset
    evaluate $DATASET
else
    echo "Dataset name not recognized."
fi
