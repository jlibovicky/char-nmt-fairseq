#!/bin/bash

set -ex

MODEL=checkpoints/ende-lee2bpe
DATA=$1
OUTPUT=$2

CHECKPOINT=$MODEL/checkpoint.avg.pt
if [ ! -e $CHECKPOINT ] || [ ${MODEL}/checkpoint_last.pt -nt $CHECKPOINT ]; then
    python scripts/average_checkpoints.py --inputs ${MODEL}/checkpoint.best_loss_*pt --output $CHECKPOINT
fi

TMPFILE=$(mktemp)

cut -c -1000 $DATA | sed -e 's/ /▁/g;s/^/▁/;s/\(.\)/\1 /g' |  > $TMPFILE

python -u fairseq_cli/interactive.py \
    $MODEL \
    --path $CHECKPOINT \
    --max-tokens 10000 \
    --unkpen 5.0 \
    --beam 10 --lenpen 1.6 \
    --source-lang en --target-lang de \
    --input $TMPFILE | grep '^H-' | sed -e 's/.*\t//' | sed 's/@@ //g' | sacremoses -l de detokenize

rm $TMPFILE
