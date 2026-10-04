#!/bin/bash

# set path to virtual environment if needed

source venv/bin/activate

## Paths
DATA_CSV="./gest_1.1.csv"
RAW_DIR="../translations/raw_translations"
FILTERED_DIR="../translations/filtered_translations"
FINAL_DIR="../translations/final_translations"

JSON_KEY="/path/to/you/json/key.json"
PROJECT_ID="your_project_id"

SAMPLE_SIZE=5 # Use 1 for full dataset, >1 for testing (££ safety)
LANGUAGES="all" # Use "all" for full languages, or specify a language e.g., "French"

THRESHOLD="0.85" # set your threshold for filtering out low-quality translations
QE_MODEL="Unbabel/wmt22-cometkiwi-da" # other QE models can be used

NUM_GENDERED_WORDS="1" # language heuristics for filtering gendered minimal pairs 
NUM_DIFFERENT_LETTERS="2"

# --- 5. Language Configuration ---
if [[ "$LANGUAGE" == "all" ]]; then
    LANGUAGES=(
        "English" "Bulgarian" "Danish" "Dutch" "Estonian" 
        "Finnish" "French" "German" "Greek" "Hungarian" "Irish" 
        "Italian" "Latvian" "Lithuanian" "Maltese" "Portuguese" "Romanian" 
        "Spanish" "Swedish" "Catalan" "Galician" "Norwegian" "Turkish" "Croatian" 
        "Czech" "Polish" "Slovak" "Slovenian" "Russian" "Ukrainian"
    )
    echo "Evaluating for all languages."
else
    echo "Evaluating for language: ${LANGUAGE}"
    LANGUAGES=("${LANGUAGE}")
fi

# ========================================= #
# ==== Step 1: TRANSLATE WITH GOOGLE API=== #
# ========================================= #

# python translation_script.py \
#     --data_path "$DATA_CSV" \
#     --json_key "$JSON_KEY" \
#     --project_id "$PROJECT_ID" \
#     --out_dir "$RAW_DIR" \
#     --sample_size $SAMPLE_SIZE \
#     --languages "${LANGUAGES[@]}"

# ========================================= #
# =====Step 2: FILTER WITH COMET QE ======= #
# ========================================= #

# python qe_filtering.py \
#     --languages "${LANGUAGES[@]}" \
#     --threshold $THRESHOLD \
#     --batch_size 16 \
#     --qe_model_name "$QE_MODEL" \
#     --data_dir "$RAW_DIR" \
#     --out_dir "$FILTERED_DIR" \

# ========================================= #
# =====Step 3: CREATE FINAL DATASET ======= #
# ========================================= #

python heuristic_filtering.py \
    --languages "${LANGUAGES[@]}" \
    --num_gendered_words $NUM_GENDERED_WORDS \
    --num_different_letters $NUM_DIFFERENT_LETTERS \
    --input_dir "$FILTERED_DIR" \
    --out_dir "$FINAL_DIR" \
    --gest_path "$DATA_CSV"
