#!/bin/bash

set -ex

python train.py data-bin/czeng-char \
    --save-dir checkpoints/czeng-char \
    --tensorboard-logdir tensorboard/czeng-char \
    --arch transformer_wmt_en_de_big_t2t --share-all-embeddings \
    --label-smoothing 0.1 \
    --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.998)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 16000 \
    --update-freq 4 \
    --max-tokens 500 \
    --max-update 10000000 \
    --keep-best-checkpoints 5 \
    --save-interval-updates 2000 \
    --keep-interval-updates 10 \
    --patience 10 \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu-print-samples \
    --max-source-positions 4096 \
    --max-target-positions 4096 \
    --num-workers 25 \
    --no-epoch-checkpoints
