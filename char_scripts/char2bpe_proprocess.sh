#!/bin/bash

set -ex

python fairseq_cli/preprocess.py --source-lang en --target-lang cs \
    --trainpref data/czeng/train.char2bpe --validpref data/czeng/val.char2bpe --testpref data/czeng/test.char2bpe \
    --destdir data-bin/czeng-char2bpe \
    --workers 20 \

python fairseq_cli/preprocess.py --source-lang en --target-lang de \
    --trainpref data/ende/train.char2bpe --validpref data/ende/val.char2bpe --testpref data/ende/test.char2bpe \
    --destdir data-bin/ende-char2bpe \
    --workers 20 \
