#!/usr/bin/env bash
bash embeddingless_scripts/train_byte_embeddingless.sh en es 14 0.3 8000
bash embeddingless_scripts/evaluate.sh byte_embeddingless en es 14 0.3 valid
bash embeddingless_scripts/score.sh byte_embeddingless en es 14 0.3 valid
