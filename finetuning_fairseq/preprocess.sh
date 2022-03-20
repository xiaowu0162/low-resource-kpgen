#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

DATA_DIR=${HOME_DIR}/data
MODEL_DIR=${HOME_DIR}/models
mkdir -p $MODEL_DIR

FB_DL_URL=https://dl.fbaipublicfiles.com/fairseq

for size in base; do   # large; do
    if [[ ! -d ${MODEL_DIR}/bart.${size} ]]; then
        wget -N ${FB_DL_URL}/models/bart.${size}.tar.gz -P $MODEL_DIR
        tar -xvzf ${MODEL_DIR}/bart.${size}.tar.gz -C $MODEL_DIR
        rm ${MODEL_DIR}/bart.${size}.tar.gz
    fi
done

GPT2_BPE_DIR=${MODEL_DIR}/gpt2_bpe
mkdir -p $GPT2_BPE_DIR

for filename in "encoder.json" "vocab.bpe" "dict.txt"; do
    if [[ ! -f ${GPT2_BPE_DIR}/${filename} ]]; then
        wget -N ${FB_DL_URL}/gpt2_bpe/${filename} -P $GPT2_BPE_DIR
    fi
done

DICT_FILE=${MODEL_DIR}/bart.base/dict.txt # dict.txt

function bpe_preprocess () {

if [[ "$TASK" =~ ^(kp20k|nus|inspec|krapivin|semeval)$ ]]; then
    IN_DIR=$DATA_DIR/scikp/$TASK/fairseq
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe
elif [[ "$TASK" =~ ^(kp20k-20k-1|kp20k-20k-2|kp20k-20k-3)$ ]]; then
    IN_DIR=$DATA_DIR/kp20k-20k/$TASK/fairseq
    OUT_DIR=$DATA_DIR/kp20k-20k/$TASK/fairseq/gpt2_bpe
else
    IN_DIR=$DATA_DIR/kptimes/fairseq
    OUT_DIR=$DATA_DIR/kptimes/fairseq/gpt2_bpe
fi

mkdir -p $OUT_DIR

for SPLIT in train valid test; do
    for LANG in source target; do
        python ${HOME_DIR}/utils/encode.py \
            --model bart \
            --encoder-json ${GPT2_BPE_DIR}/encoder.json \
            --vocab-bpe ${GPT2_BPE_DIR}/vocab.bpe \
            --inputs $IN_DIR/$SPLIT.$LANG \
            --outputs $OUT_DIR/$SPLIT.bpe.$LANG \
            --max_len 510 \
            --workers 60;
    done
done

}

function process () {

if [[ "$TASK" =~ ^(kp20k|nus|inspec|krapivin|semeval)$ ]]; then
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe
elif [[ "$TASK" =~ ^(kp20k-20k-1|kp20k-20k-2|kp20k-20k-3)$ ]]; then
    OUT_DIR=$DATA_DIR/kp20k-20k/$TASK/fairseq/gpt2_bpe
else
    OUT_DIR=$DATA_DIR/kptimes/fairseq/gpt2_bpe
fi

fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --trainpref $OUT_DIR/train.bpe \
    --validpref $OUT_DIR/valid.bpe \
    --testpref $OUT_DIR/test.bpe \
    --destdir $OUT_DIR/binary \
    --workers 60 \
    --srcdict $DICT_FILE \
    --tgtdict $DICT_FILE;

}

function bpe_preprocess_test_only () {

IN_DIR=$DATA_DIR/scikp/$TASK/fairseq
OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe

mkdir -p $OUT_DIR

for SPLIT in test; do
  for LANG in source target; do
    python ${HOME_DIR}/utils/encode.py \
        --model bart \
        --encoder-json ${GPT2_BPE_DIR}/encoder.json \
        --vocab-bpe ${GPT2_BPE_DIR}/vocab.bpe \
        --inputs $IN_DIR/$SPLIT.$LANG \
        --outputs $OUT_DIR/$SPLIT.bpe.$LANG \
        --max_len 510 \
        --workers 60;
  done
done

}

function process_test_only () {

OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe

fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --testpref $OUT_DIR/test.bpe \
    --destdir $OUT_DIR/binary \
    --workers 60 \
    --srcdict $DICT_FILE \
    --tgtdict $DICT_FILE;

}


for task in kp20k kp20k-20k-1 kp20k-20k-2 kp20k-20k-3; do
    TASK=$task
    bpe_preprocess
    process
done
for task in nus inspec krapivin semeval; do
    TASK=$task
    bpe_preprocess_test_only
    process_test_only
done
