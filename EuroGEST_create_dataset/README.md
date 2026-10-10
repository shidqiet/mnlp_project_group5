# EuroGEST: Dataset Creation

This repository contains the pipeline for translating English source sentences (e.g., from GEST) into 29 European languages using the Google Cloud Translation API and detecting gender marking in the target languages. 

## Pipeline overview
1. translation_script.py: Translates sentences. For gender-neutral languages, it translates directly. For gendered languages, it uses "forced-gender" wrappers ("The man said...") to elicit masculine and feminine forms.
2. qe_filtering.py: Uses a machine translation quality estimation model to remove translations that fall below a quality threshold.
3. heuristic_filtering.py: Analyzes masculine/feminine pairs to categorize them as identical = Neutral, minimally different = Gendered Minimal Pairs, or Unknown. Minimally different can be defined with a set number of possible different words and letters. 

The original gest data is included in a .csv file for tranlsation. The outputs from the pipeline save in a 'translations' folder ('raw_translations', 'filtered_translations', and 'final_translations'). 

## Setup Instructions

**1. Setup Environment** 
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Cloud Configuration** 

You need a Google Cloud Project with the Cloud Translation API enabled.
- Download your Service Account JSON key and set these variables
- Update the json key path and project ID variables inside translate_dataset.sh
- Note: Never commit your key.json file to GitHub. It's included in the .gitignore.

**3. Parameter setting** 
Adjust the variables in the .sh script as required (e.g. QE Threshold, NUM_GENDERED_WORDS and NUM_DIFFERENT_LETTERS for minimal pair detection, languages etc.) 
Set the pathnames in the .sh script to where you want to save your translated data. 

**4. Testing & Execution** 
Translation API usage incurs costs. Always test on a small sample first.
To Test: Set SAMPLE_SIZE=5 in the .sh script.
To Run Full Pipeline: Set SAMPLE_SIZE=1.

```bash
./translate_dataset.sh
```
## QE scoring

`qe_scoring.py` adds a COMET-QE score to every translated sentence. It runs after the translation
script and before `heuristic_filtering.py`.

Unlike `qe_filtering.py`, which drops every row at or below the threshold and keeps no scores,
nothing is dropped here: each row keeps its score, so the threshold can be chosen - and changed -
afterwards.

```bash
./score_dataset.sh
```

or directly:

```bash
python qe_scoring.py \
    --languages "Chinese,Indonesian" \
    --data_dir ./translated_output \
    --out_dir ./scored_output \
    --gest_path ./gest_1.1.csv
```

**Input**: the files written by the translation scripts (row index = GEST row, `GEST_sentence`,
`translation`, `original_stereotype`).
**Output**: `<out_dir>/<language>.csv`, the same rows and columns with `COMET_score` added, plus
`qe_summary.csv` with per-language min/median/mean/max and how many rows are above the threshold.

Notes:

- The model (`Unbabel/wmt22-cometkiwi-da`) is **gated** on Hugging Face. Accept the conditions on the
  model page, then log in once with `huggingface-cli login`. The first run downloads 2.3 GB.
- The checkpoint is pinned by revision, and scores are cached in `<out_dir>/.qe_cache.sqlite`, so a
  rerun costs nothing and returns exactly the same numbers. Delete that file to force rescoring.
- Every row is checked against `gest_1.1.csv` (row number in range, `GEST_sentence` and
  `original_stereotype` equal to GEST). The run stops if a file no longer lines up, instead of
  quietly scoring sentences against the wrong source.
- `requirements.txt` pins `torch==2.9.1`, which needs **Python 3.10 or newer**. On 3.9 pip resolves
  older torch and COMET, and the scores can differ.
- Rows with an empty translation keep an empty score, so an interrupted translation run can be
  scored and finished later.
- This script scores gender-neutral languages (one translation per sentence). For gendered languages,
  `qe_filtering.py` scores the masculine and feminine templates.
