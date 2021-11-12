#!/bin/bash

set -ex

MODEL=checkpoints/ende-lee2bpe
DATA=data/ende/val.en

CHECKPOINT=$MODEL/checkpoint.avg.pt
if [ ! -e $CHECKPOINT ] || [ ${MODEL}/checkpoint_last.pt -nt $CHECKPOINT ]; then
    python scripts/average_checkpoints.py --inputs ${MODEL}/checkpoint.best_loss_*pt --output $CHECKPOINT
fi

TMPFILE=$(mktemp)

sed -e 's/ /▁/g;s/^/▁/;s/\(.\)/\1 /g' $DATA > $TMPFILE

for NORM in 0.{0..8..2} 1.{0..6..2}; do
    python -u fairseq_cli/interactive.py \
        $MODEL \
        --path $CHECKPOINT \
        --max-tokens 10000 \
        --unkpen 5.0 \
        --beam 10 --lenpen $NORM \
        --source-lang en --target-lang de \
        --input $TMPFILE | grep '^H-' | sed -e 's/.*\t//' | sed 's/@@ //g' | sacremoses -l de detokenize > grid_beam/ende/lee2bpe/${NORM}
done

rm $TMPFILE
