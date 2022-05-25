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


# Please make sure to run preprocess_ssr.sh first as it calls agg_data.py and does some basic preprocessing that this script depends on.

TASK=kp20k-salient-span


#################################################################################
# Option 1: ssp-m

RAW_BPE_DIR=$DATA_DIR/scikp/kp20k-salient-span/fairseq/gpt2_bpe/
MASKED_BPE_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssr-m_in_order/
mkdir -p ${MASKED_BPE_DIR}
OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssp-m/

N_SHARDS=35
SHUFFLE=False
python prepare_ssr.py kp20k ${N_SHARDS} ${RAW_BPE_DIR} ${MASKED_BPE_DIR} ${SHUFFLE}

# move things around
for s in `seq 1 ${N_SHARDS}`; do
    mkdir -p ${MASKED_BPE_DIR}/shard${s}
    mv ${MASKED_BPE_DIR}/train.bpe.source.shard${s} ${MASKED_BPE_DIR}/shard${s}/train.bpe.source
    mv ${MASKED_BPE_DIR}/train.bpe.target.shard${s} ${MASKED_BPE_DIR}/shard${s}/train.bpe.target
done
mv ${MASKED_BPE_DIR}/valid.bpe.* ${MASKED_BPE_DIR}/test.bpe.* ${MASKED_BPE_DIR}/shard1


# get all bpe files
for s in `seq 1 ${N_SHARDS}`; do
    echo "Shard ${s}"
    mkdir -p ${OUT_DIR}/shard${s}
    # shuffle and dedup
    python shuffle_and_deduplify_for_ssp.py ${MASKED_BPE_DIR}/shard${s}/train.bpe.source ${RAW_BPE_DIR}/train.bpe.target ${OUT_DIR}/shard${s}
done
cp ${MASKED_BPE_DIR}/shard1/valid.bpe.source ${MASKED_BPE_DIR}/shard1/test.bpe.source ${OUT_DIR}/shard1
cp ${RAW_BPE_DIR}/valid.bpe.target ${RAW_BPE_DIR}/test.bpe.target ${OUT_DIR}/shard1

# binarize
process_shards

# remove the intermediate files
rm -r ${MASKED_BPE_DIR}

#################################################################################
# Option 2: ssp-d

RAW_BPE_DIR=$DATA_DIR/scikp/kp20k-salient-span/fairseq/gpt2_bpe/
MASKED_BPE_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssr-d_in_order/
mkdir -p ${MASKED_BPE_DIR}
OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/gpt2_bpe_ssp-d/

N_SHARDS=35
SHUFFLE=False
python prepare_ssr_deletion.py kp20k ${N_SHARDS} ${RAW_BPE_DIR} ${MASKED_BPE_DIR} ${SHUFFLE} 

# move things around
for s in `seq 1 ${N_SHARDS}`; do
    mkdir -p ${MASKED_BPE_DIR}/shard${s}
    mv ${MASKED_BPE_DIR}/train.bpe.source.shard${s} ${MASKED_BPE_DIR}/shard${s}/train.bpe.source
    mv ${MASKED_BPE_DIR}/train.bpe.target.shard${s} ${MASKED_BPE_DIR}/shard${s}/train.bpe.target
done
mv ${MASKED_BPE_DIR}/valid.bpe.* ${MASKED_BPE_DIR}/test.bpe.* ${MASKED_BPE_DIR}/shard1


# get all bpe files
for s in `seq 1 ${N_SHARDS}`; do
    echo "Shard ${s}"
    mkdir -p ${OUT_DIR}/shard${s}
    # shuffle and dedup 
    python shuffle_and_deduplify_for_ssp.py ${MASKED_BPE_DIR}/shard${s}/train.bpe.source ${RAW_BPE_DIR}/train.bpe.target ${OUT_DIR}/shard${s}
done
cp ${MASKED_BPE_DIR}/shard1/valid.bpe.source ${MASKED_BPE_DIR}/shard1/test.bpe.source ${OUT_DIR}/shard1
cp ${RAW_BPE_DIR}/valid.bpe.target ${RAW_BPE_DIR}/test.bpe.target ${OUT_DIR}/shard1

# binarize
process_shards

# remove the intermediate files
rm -r ${MASKED_BPE_DIR}
