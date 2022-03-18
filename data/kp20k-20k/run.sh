#!/usr/bin/env bash

SRC_DIR=`realpath ..`;


function faiseq_prepare () {
    for subset in 1 2 3; do
        outdir=kp20k-20k-${subset}/fairseq
        mkdir -p $outdir
        for split in train valid test; do
        PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
                -input_json kp20k-20k-${subset}/processed/${split}.json \
                -output_source $outdir/${split}.source \
                -output_target $outdir/${split}.target;
        done
    done
}


function json_prepare () {
    for subset in 1 2 3; do   
        outdir=kp20k-20k-${subset}/json
        mkdir -p $outdir
        for split in train valid test; do
        PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
                -format json \
                -input_json kp20k-20k-${subset}/processed/${split}.json \
                -output_source $outdir/${split}.json;
        done
    done
}


function hf_json_prepare () {
    for subset in 1 2 3; do   
        outdir=kp20k-20k-${subset}/json_hf
        mkdir -p $outdir
        for split in train valid test; do
        PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
                -format hf_json \
                -input_json kp20k-20k-${subset}/processed/${split}.json \
                -output_source $outdir/${split}.json;
        done
    done
}


faiseq_prepare
json_prepare
hf_json_prepare

