#!/bin/bash

set -ex

MODEL=checkpoints/czeng-lee2bpe
DATA=$1
OUTPUT=$2

CHECKPOINT=$MODEL/checkpoint.avg.pt
if [ ! -e $CHECKPOINT ] || [ ${MODEL}/checkpoint_last.pt -nt $CHECKPOINT ]; then
    python scripts/average_checkpoints.py --inputs ${MODEL}/checkpoint.best_loss_*pt --output $CHECKPOINT
fi

TMPFILE=$(mktemp)

cut -c -1000 $DATA | sed -e 's/ /▁/g;s/^/▁/;s/\(.\)/\1 /g' > $TMPFILE

python -u fairseq_cli/interactive.py \
    $MODEL \
    --path $CHECKPOINT \
    --max-tokens 10000 \
    --unkpen 5.0 \
    --beam 10 --lenpen 1.0 \
    --source-lang en --target-lang cs \
    --input $TMPFILE | grep '^H-' | sed -e 's/.*\t//' | sed 's/@@ //g' | sacremoses -l cs detokenize

rm $TMPFILE
