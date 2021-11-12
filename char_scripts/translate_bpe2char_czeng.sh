#!/bin/bash

set -ex

MODEL=checkpoints/czeng-bpe2char
DATA=$1
OUTPUT=$2

CHECKPOINT=$MODEL/checkpoint.avg.pt
if [ ! -e $CHECKPOINT ] || [ ${MODEL}/checkpoint_last.pt -nt $CHECKPOINT ]; then
    python scripts/average_checkpoints.py --inputs ${MODEL}/checkpoint.best_loss_*pt --output $CHECKPOINT
fi

TMPFILE=$(mktemp)

sacremoses -l en tokenize < $DATA | subword-nmt apply-bpe --codes data/czeng/bpe16k > $TMPFILE

python -u fairseq_cli/interactive.py \
    $MODEL \
    --path $CHECKPOINT \
    --max-tokens 10000 \
    --unkpen 5.0 \
    --beam 10 --lenpen 1.2 \
    --max-len-a 0 --max-len-b 1024 \
    --source-lang en --target-lang cs \
    --input $TMPFILE | grep '^H-' | sed -e 's/.*\t//' | sed 's/ //g;s/▁/ /g'

rm $TMPFILE
