#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../..`;

DATA_BASE_DIR=${HOME_DIR}/data
OUT_DIR=${CURRENT_DIR}/data
mkdir -p $OUT_DIR

DATA_DIR=${DATA_BASE_DIR}/scikp/kp20k_separated

for split in train valid test; do
    if [[ ! -f $OUT_DIR/kp20k.${split}.jsonl ]]; then
        python ${CURRENT_DIR}/source/convert.py \
            -src_file $DATA_DIR/${split}_src.txt \
            -tgt_file $DATA_DIR/${split}_trg.txt \
            -out_file $OUT_DIR/kp20k.${split}.jsonl \
            -dataset kp20k \
            -split $split;
    fi
done
