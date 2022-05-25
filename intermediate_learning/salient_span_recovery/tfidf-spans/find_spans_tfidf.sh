#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

python -m spacy download en

out_df_dir=${CURRENT_DIR}/tfidf/raw_preds
mkdir -p ${out_df_dir}


for split in train valid test; do
    echo "Converting kp20k-${split} to xml"
    in_json_file=${HOME_DIR}/data/scikp/kp20k/processed/${split}.json
    out_xml_dir=${CURRENT_DIR}/tfidf/data_${split}/
    mkdir -p ${out_xml_dir}
    python convert_json_to_xml.py ${in_json_file} ${out_xml_dir}
done


echo "Counting document frequency"
python compute_document_frequency_folder.py --input tfidf/data_train --output ${out_df_dir} -n 3 --stopwords --tags description


for split in train valid test; do
    echo "Running tf-idf on kp20k-${split}"
    python run_tfidf.py --input tfidf/data_${split} --model tfidf --language en --output tfidf/raw_preds/results_tfidf_kp20k_${split}.csv -n 3 -k 30 --stopwords --tags description --redundancy_removal True
done


mkdir -p tfidf/merged
for split in train valid test; do
    echo "Collecting tf-idf predictions on kp20k-${split}"
    python collect_tfidf_predictions.py \
        tfidf/raw_preds/results_tfidf_kp20k_${split}.csv \
        ${HOME_DIR}/data/scikp/kp20k/processed/${split}.json \
        tfidf/merged/kp20k.${split}.tfidfpred.jsonl
done

