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
