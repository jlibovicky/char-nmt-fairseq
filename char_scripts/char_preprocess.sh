#python fairseq_cli/preprocess.py --source-lang en --target-lang cs \
#    --trainpref data/czeng/train --validpref data/czeng/val --testpref data/czeng/test \
#    --destdir data-bin/czeng-char \
#    --workers 20 \
#    --joined-dictionary \
#    --char-tokens

#python fairseq_cli/preprocess.py --source-lang en --target-lang de \
#    --trainpref data/ende/train --validpref data/ende/val --testpref data/ende/test \
#    --destdir data-bin/ende-char \
#    --workers 20 \
#    --joined-dictionary \
#    --char-tokens

python fairseq_cli/preprocess.py --source-lang en --target-lang cs \
    --trainpref data/czeng/train.dummy --validpref data/czeng/val --testpref data/czeng/test \
    --destdir data-bin/czeng-dummy \
    --workers 20 \
    --joined-dictionary \
    --char-tokens

