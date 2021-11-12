#!/bin/bash

set -ex

SCRIPT=translate_lee2bpe_czeng.sh
DIRECTION=encs
TYPE=lee2bpe

# WMT20 <<<<<
bash char_scripts/$SCRIPT  data/${DIRECTION}/test.en > outputs/${DIRECTION}/wmt20/${TYPE}.txt

# IT <<<<<<
bash char_scripts/$SCRIPT testsets/${DIRECTION}_it/it_en.txt > outputs/${DIRECTION}/it/${TYPE}.txt

# NHS <<<<<
bash char_scripts/$SCRIPT testsets/${DIRECTION}_nhs/nhs_en.txt > outputs/${DIRECTION}/medical/${TYPE}.txt

# GENDER TEST SET <<<<<
bash char_scripts/$SCRIPT testsets/gender/en.txt > outputs/${DIRECTION}/gender/${TYPE}.txt

# MORPHEVAL <<<<<
bash char_scripts/$SCRIPT testsets/morpheval/morpheval.limsi.v2.en.sents > outputs/${DIRECTION}/morpheval/${TYPE}.txt

# TEXTFLINT <<<<<
for F in testsets/textflint/encs_*; do
    bash char_scripts/$SCRIPT $F > outputs/${DIRECTION}/textflint/${TYPE}_${F:24:100}
done
