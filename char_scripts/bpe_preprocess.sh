python fairseq_cli/preprocess.py --source-lang en --target-lang cs \
    --trainpref data/czeng/train.bpe --validpref data/czeng/val.bpe --testpref data/czeng/test.bpe \
    --destdir data-bin/czeng-bpe \
    --workers 20 \
    --joined-dictionary

python fairseq_cli/preprocess.py --source-lang en --target-lang de \
    --trainpref data/ende/train.bpe --validpref data/ende/val.bpe --testpref data/ende/test.bpe \
    --destdir data-bin/ende-bpe \
    --workers 20 \
    --joined-dictionary
