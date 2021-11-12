python fairseq_cli/preprocess.py --source-lang en --target-lang cs \
    --trainpref data/czeng/train.bpe2char --validpref data/czeng/val.bpe2char --testpref data/czeng/test.bpe2char \
    --destdir data-bin/czeng-bpe2char \
    --workers 20 \

python fairseq_cli/preprocess.py --source-lang en --target-lang de \
    --trainpref data/ende/train.bpe2char --validpref data/ende/val.bpe2char --testpref data/ende/test.bpe2char \
    --destdir data-bin/ende-bpe2char \
    --workers 20 \
