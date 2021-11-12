#!/bin/bash

set -ex

python train.py data-bin/czeng-bpe \
    --save-dir checkpoints/czeng-bpe \
    --tensorboard-logdir tensorboard/czeng-bpe \
    --arch transformer_wmt_en_de_big_t2t --share-all-embeddings \
    --label-smoothing 0.1 \
    --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.998)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 16000 \
    --max-tokens 1800 \
    --max-update 1000000 \
    --keep-best-checkpoints 5 \
    --save-interval-updates 2000 \
    --keep-interval-updates 10 \
    --patience 10 \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu-print-samples \
    --num-workers 25 \
    --no-epoch-checkpoints
