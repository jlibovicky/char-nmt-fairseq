#!/usr/bin/env bash
bash embeddingless_scripts/train_byte.sh en ja 17 0.3 8000
bash embeddingless_scripts/evaluate.sh byte en ja 17 0.3 valid
bash embeddingless_scripts/score.sh byte en ja 17 0.3 valid
