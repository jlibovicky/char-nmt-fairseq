#!/usr/bin/env bash
bash embeddingless_scripts/train_byte.sh ja en 17 0.3 8000
bash embeddingless_scripts/evaluate.sh byte ja en 17 0.3 valid
bash embeddingless_scripts/score.sh byte ja en 17 0.3 valid
