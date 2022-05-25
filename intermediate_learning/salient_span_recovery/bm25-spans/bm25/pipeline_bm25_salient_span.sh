#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_BASE_DIR=`realpath ../..`;


# start the server first by running
# bash CLIReval/scripts/server.sh start

DATASETS=(
    kp20k-train
    kp20k-valid
    kp20k-test
)

DATASET_NAME=${1:-kp20k-train};

if [[ ! " ${DATASETS[@]} " =~ " $DATASET_NAME " ]]; then
    echo "Dataset name must be from [$(IFS=\| ; echo "${DATASETS[*]}")].";
    echo "bash retrieve.sh <dataset> <keyword-type>";
    exit;
fi

DATA_DIR=${CODE_BASE_DIR}/retrieval/data;
OUT_DIR=$CURRENT_DIR/raw_preds/
export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;

FILES=()
if [[ $DATASET_NAME == "kp20k-train" ]]; then
    FILES+=(${DATA_DIR}/kp20k.train.jsonl)
    DOMAIN=kp20k-train
    SUBSET=train
    DB_PATH="${DATA_DIR}/${DOMAIN}.db";
elif [[ $DATASET_NAME == "kp20k-valid" ]]; then
    FILES+=(${DATA_DIR}/kp20k.train.jsonl)
    FILES+=(${DATA_DIR}/kp20k.valid.jsonl)
    DOMAIN=kp20k-valid
    SUBSET=valid
    DB_PATH="${DATA_DIR}/${DOMAIN}.db";
elif [[ $DATASET_NAME == "kp20k-test" ]]; then
    FILES+=(${DATA_DIR}/kp20k.train.jsonl)
    FILES+=(${DATA_DIR}/kp20k.valid.jsonl)
    FILES+=(${DATA_DIR}/kp20k.test.jsonl)
    DOMAIN=kp20k-test
    SUBSET=test
    DB_PATH="${DATA_DIR}/${DOMAIN}.db";
fi

# Create db from preprocessed data.
if [[ ! -f $DB_PATH ]]; then
    python build_db.py --files "${FILES[@]}" --save_path $DB_PATH;
fi

# Index the preprocessed documents.
python build_es.py --db_path $DB_PATH --domain $DOMAIN --config_file_path bm25_config.json --port 9200;

# Search documents based on BM25 scores.
NDOCS=1000

OUTFILE=${OUT_DIR}/kp20k-${SUBSET}-bm25-salient${NDOCS}.json
python es_search_salient_span.py \
    --index_name ${DOMAIN}_search_test \
    --input_data_file ${DATA_DIR}/kp20k.${SUBSET}.jsonl \
    --output_fp $OUTFILE \
    --keyword $KEYWORD_TYPE \
    --n_docs $NDOCS \
    --port 9200;

