#!/bin/bash

set -ex

MODEL=checkpoints/ende-char
DATA=data/ende/val.en

CHECKPOINT=$MODEL/checkpoint.avg.pt
if [ ! -e $CHECKPOINT ] || [ ${MODEL}/checkpoint_last.pt -nt $CHECKPOINT ]; then
    python scripts/average_checkpoints.py --inputs ${MODEL}/checkpoint.best_loss_*pt --output $CHECKPOINT
fi

TMPFILE=$(mktemp)

cut -c -1024 $DATA > $TMPFILE

for NORM in 0.{0..8..2} 1.{0..6..2}; do
    python -u fairseq_cli/interactive.py \
        $MODEL \
        --path $CHECKPOINT \
        --max-tokens 10000 \
        --unkpen 5.0 \
        --beam 10 --lenpen $NORM \
        --source-lang en --target-lang de \
        --char-tokens \
        --max-len-a 0 --max-len-b 1024 \
        --input $TMPFILE | grep '^H-' | sed -e 's/.*\t//' > grid_beam/ende/char/${NORM}

done

rm $TMPFILE
