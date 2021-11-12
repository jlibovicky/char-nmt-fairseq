#!/bin/bash

set -ex

MODEL=checkpoints/czeng-char
DATA=$1
OUTPUT=$2

CHECKPOINT=$MODEL/checkpoint.avg.pt
if [ ! -e $CHECKPOINT ] || [ ${MODEL}/checkpoint_last.pt -nt $CHECKPOINT ]; then
    python scripts/average_checkpoints.py --inputs ${MODEL}/checkpoint.best_loss_*pt --output $CHECKPOINT
fi

TMPFILE=$(mktemp)

cut -c -1023 $DATA > $TMPFILE

python -u fairseq_cli/interactive.py \
    $MODEL \
    --path $CHECKPOINT \
    --max-tokens 10000 \
    --unkpen 5.0 \
    --beam 10 --lenpen 1.4 \
    --source-lang en --target-lang cs \
    --char-tokens \
    --max-len-a 0 --max-len-b 1024 \
    --input $TMPFILE | grep '^H-' | sed -e 's/.*\t//'

rm $TMPFILE
