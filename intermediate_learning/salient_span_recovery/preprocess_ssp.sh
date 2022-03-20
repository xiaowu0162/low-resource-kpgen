#!/usr/bin/env bash

# Please make sure to run `preprocess_ssr.sh` first. 
# This script just put together the outputs from `preprocess_ssr.sh`.

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

TASK=kp20k-salient-span
RAW_BPE_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe/
MASKED_BPE_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssr/
OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssp/

# change this number according to the ssr settings
N_SHARDS=35


function process_shards () {

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--trainpref "${OUT_DIR}/shard1/train.bpe" \
--validpref "${OUT_DIR}/shard1/valid.bpe" \
--testpref "${OUT_DIR}/shard1/test.bpe" \
--destdir "${OUT_DIR}/binary/shard1" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

for shard in `seq 2 ${N_SHARDS}`; do 
    fairseq-preprocess \
	--source-lang "source" \
	--target-lang "target" \
	--trainpref "${OUT_DIR}/shard${shard}/train.bpe" \
	--destdir "${OUT_DIR}/binary/shard${shard}" \
	--workers 60 \
	--srcdict $DICT_FILE \
	--tgtdict $DICT_FILE;
done

}


# get all bpe files
for s in `seq 1 ${N_SHARDS}`; do
    echo "Shard ${s}"
    mkdir -p ${OUT_DIR}/shard${s}
    cp ${MASKED_BPE_DIR}/shard${s}/train.bpe.source ${OUT_DIR}/shard${s}
    cp ${RAW_BPE_DIR}/train.bpe.target ${OUT_DIR}/shard${s}
done
cp ${MASKED_BPE_DIR}/shard1/valid.bpe.source ${MASKED_BPE_DIR}/shard1/test.bpe.source ${OUT_DIR}/shard1
cp ${RAW_BPE_DIR}/valid.bpe.target ${RAW_BPE_DIR}/test.bpe.target ${OUT_DIR}/shard1

# binarize
process_shards
