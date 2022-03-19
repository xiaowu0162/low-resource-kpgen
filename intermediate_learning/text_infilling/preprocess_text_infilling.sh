#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

# SRCDIR=data
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

DICT_FILE=${MODEL_DIR}/bart.base/dict.txt 


function bpe_preprocess () {

if [[ "$TASK" =~ ^(kp20k-lm)$ ]]; then
    IN_DIR=$DATA_DIR/scikp/$TASK/fairseq
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe
fi

mkdir -p $OUT_DIR

for SPLIT in train valid test; do
    python ${HOME_DIR}/utils/encode.py \
        --model bart \
        --encoder-json ${GPT2_BPE_DIR}/encoder.json \
        --vocab-bpe ${GPT2_BPE_DIR}/vocab.bpe \
        --inputs $IN_DIR/$SPLIT \
        --outputs $OUT_DIR/$SPLIT.bpe \
        --max_len 510 \
        --workers 60;
done

}


function process () {

if [[ "$TASK" =~ ^(kp20k-lm)$ ]]; then
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe
fi

fairseq-preprocess \
    --task "denoising" \
    --trainpref $OUT_DIR/train.bpe \
    --validpref $OUT_DIR/valid.bpe \
    --testpref $OUT_DIR/test.bpe \
    --destdir $OUT_DIR/binary \
    --workers 60 \
    --srcdict $DICT_FILE \
    --tgtdict $DICT_FILE;

}


# aggregate data first
python agg_data.py

for task in kp20k-lm; do
   TASK=$task
   bpe_preprocess
   process

   # move things around
   for s in train valid test; do
       mv $DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe/binary/${s}.None-None.bin $DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe/binary/${s}.bin
       mv $DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe/binary/${s}.None-None.idx $DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe/binary/${s}.idx
   done
done
