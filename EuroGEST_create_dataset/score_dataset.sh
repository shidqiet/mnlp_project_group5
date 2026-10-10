#!/bin/bash

# Add COMET-QE scores to the translated files. Run this after the translation script and before
# heuristic_filtering.py. No rows are dropped: every row keeps its score.

# set path to virtual environment if needed

source venv/bin/activate

## Paths
GEST_CSV="./gest_1.1.csv"
TRANSLATED_DIR="./translated_output"
SCORED_DIR="./scored_output"

LANGUAGES="Chinese,Indonesian"

BATCH_SIZE=16
THRESHOLD="0.85"   # only reported, never applied
QE_MODEL="Unbabel/wmt22-cometkiwi-da"

python qe_scoring.py \
    --languages "$LANGUAGES" \
    --data_dir "$TRANSLATED_DIR" \
    --out_dir "$SCORED_DIR" \
    --gest_path "$GEST_CSV" \
    --batch_size $BATCH_SIZE \
    --threshold $THRESHOLD \
    --qe_model_name "$QE_MODEL"
