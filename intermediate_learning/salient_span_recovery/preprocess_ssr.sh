#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

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

if [[ "$TASK" =~ ^(kp20k-salient-span)$ ]]; then
    IN_DIR=$DATA_DIR/scikp/$TASK/fairseq
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe
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


function process_shards () {

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--trainpref "${MASKED_BPE_DIR}/shard1/train.bpe" \
--validpref "${MASKED_BPE_DIR}/shard1/valid.bpe" \
--testpref "${MASKED_BPE_DIR}/shard1/test.bpe" \
--destdir "${MASKED_BPE_DIR}/binary/shard1" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

for shard in `seq 2 ${N_SHARDS}`; do 
    fairseq-preprocess \
	--source-lang "source" \
	--target-lang "target" \
	--trainpref "${MASKED_BPE_DIR}/shard${shard}/train.bpe" \
	--destdir "${MASKED_BPE_DIR}/binary/shard${shard}" \
	--workers 60 \
	--srcdict $DICT_FILE \
	--tgtdict $DICT_FILE;
done

}


# aggregate data first
python agg_data.py

# bpe encode the text and spans
TASK=kp20k-salient-span
bpe_preprocess

# generate masking shards in bpe
N_SHARDS=35
RAW_BPE_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe/
MASKED_BPE_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssr/
mkdir -p ${MASKED_BPE_DIR}
python prepare_ssr.py kp20k ${N_SHARDS} ${RAW_BPE_DIR} ${MASKED_BPE_DIR}

# move things around
for s in `seq 1 ${N_SHARDS}`; do
    mkdir -p ${MASKED_BPE_DIR}/shard${s}
    mv ${MASKED_BPE_DIR}/train.bpe.source.shard${s} ${MASKED_BPE_DIR}/shard${s}/train.bpe.source
    mv ${MASKED_BPE_DIR}/train.bpe.target.shard${s} ${MASKED_BPE_DIR}/shard${s}/train.bpe.target
done
mv ${MASKED_BPE_DIR}/valid.bpe.* ${MASKED_BPE_DIR}/test.bpe.* ${MASKED_BPE_DIR}/shard1

# binarize
process_shards
