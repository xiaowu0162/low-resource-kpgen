#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../../`;

out_df_dir=${CURRENT_DIR}/bm25/raw_preds
mkdir -p ${out_df_dir}

# convert the raw dataset into the proper form
bash prepare.sh

# initialize the BM25 tool
cd bm25/
git clone https://github.com/ssun32/CLIReval.git
cd CLIReval/
pip install -r requirements.txt
bash scripts/install_external_tools.sh

# start server
bash scripts/server.sh start
sleep 20s
cd ../

# start client and run retrieval
for split in train valid test; do
    bash pipeline_bm25_salient_span.sh kp20k-${split}
done

# convert the resulting spans
mkdir -p merged
python merge_predictions.py

# stop the server
cd CLIReval/
bash scripts/server.sh stop
cd ../../
