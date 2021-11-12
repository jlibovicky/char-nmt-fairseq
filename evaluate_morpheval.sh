#!/bin/bash

set -ex

TGT=$1
DATA=$2

DATA=$(realpath $DATA)

sacremoses -l $TGT tokenize < $DATA > ${DATA}.tok

if [[ $TGT == "de" ]]; then
    cd morpheval/SMOR
    echo $PWD
    ls ../..
    tr ' ' '\n' < ${DATA}.tok | sort | uniq | ./smor > ${DATA}.smored
    cd ../..
    python3 morpheval/morpheval_v2/evaluate_de.py -i ${DATA}.tok -n morpheval/morpheval.limsi.v2.en.info -d ${DATA}.smored | \
        tee ${DATA}.analysis
fi

if [[ $TGT == "cs" ]]; then
    sed 's/$/\n/' ${DATA}.tok | tr ' ' '\n' | \
        morpheval/morphodita-1.3.0-bin/bin-linux64/run_morpho_analyze \
            --input=vertical --output=vertical \
            morpheval/czech-morfflex-pdt-131112/czech-morfflex-131112.dict 1  > ${DATA}.morphodita
    python3 morpheval/morpheval_v2/evaluate_cs.py -i ${DATA}.morphodita -n morpheval/morpheval.limsi.v2.en.info | \
        tee ${DATA}.analysis
fi
