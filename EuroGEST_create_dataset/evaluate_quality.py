#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:26:02 2025

"""

from pathlib import Path
import subprocess
import pandas as pd
from comet import download_model, load_from_checkpoint
import torch
import numpy as np


# ======================================================== #
# ===================== DEVICE SETUP ===================== #
 # ======================================================= #

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Apple Metal GPU acceleration
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA GPU if available
else:
    DEVICE = torch.device("cpu") 

print(f"Using device: {DEVICE}")

# ========================================================== #
# ================== PATH HANDLING ========================= #
# ========================================================== #

def get_git_root():
    """Return the root directory of the git repository."""
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    )

GIT_REPO_PATH = get_git_root()
RAW_DIR = Path("XXX")
STEREOTYPES = range(1, 17)

# where to save your filtered data 
FILTERED_DIR = "XXX"
FILERED_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.86
COMET_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"

# ========================================================= #
# ===================== COMET MODEL ======================= #
# ========================================================= #

model_path = download_model(COMET_MODEL_NAME)
comet_model = load_from_checkpoint(model_path)
comet_model.eval()

# ===================== UTILITY FUNCTIONS ===================== #
def score_translations(sources, hypotheses, model=comet_model, batch_size=16, gpus=1):
    samples = [{"src": s, "mt": t} for s, t in zip(sources, hypotheses)]
    outputs = model.predict(
        samples=samples,
        batch_size=batch_size,
        gpus=gpus,
        accelerator="auto"
    )
    system_score, segments_scores = outputs.system_score, outputs.scores
    torch.cuda.empty_cache()
    return system_score, segments_scores

def highlight_bad_translations(sentence_scores, sources, targets, threshold=THRESHOLD):
    indexes_to_remove = []
    for i, score in enumerate(sentence_scores):
        if score <= threshold:
            print(f"Bad translation (score={score}):")
            print(f"Source: {sources[i]}")
            print(f"Target: {targets[i]}")
            indexes_to_remove.append(i)
    return indexes_to_remove

# ===================== MAIN LOOP ===================== #
for stereotype_number in STEREOTYPES:
    print(f"Processing stereotype {stereotype_number}")

    # Load raw translations
    neutral_path = FILTERED_DIR.parent / f"neutral_stereotype_{stereotype_number}.csv"
    gendered_path = FILTERED_DIR.parent / f"gendered_stereotype_{stereotype_number}.csv"

    neutral_translations = pd.read_csv(neutral_path, index_col=0)
    gendered_translations = pd.read_csv(gendered_path, header=[0,1], index_col=0)

    # ---------- Process NEUTRAL LANGUAGES ----------
    gest_sentences = list(neutral_translations["GEST_sentence"])
    system_scores_neutral = {}
    sentence_scores_neutral = {}

    for lang in neutral_translations.columns:
        if lang == "GEST_sentence":
            continue
        translated = list(neutral_translations[lang])
        sys_score, sent_scores = score_translations(gest_sentences, translated)
        system_scores_neutral[lang] = sys_score
        sentence_scores_neutral[lang] = sent_scores
        idxs_to_remove = highlight_bad_translations(sent_scores, gest_sentences, translated)
        neutral_translations.loc[neutral_translations.index[idxs_to_remove], lang] = ''

    # ---------- Process GENDERED LANGUAGES ----------
    gest_sentences = list(gendered_translations[("GEST_sentence", "Unnamed: 1_level_1")])
    sources_masc = [f"The man said: {s}" for s in gest_sentences]
    sources_fem = [f"The woman said: {s}" for s in gest_sentences]

    system_scores_masc = {}
    system_scores_fem = {}
    sentence_scores_masc = {}
    sentence_scores_fem = {}

    last_lang = None
    for (lang, condition) in gendered_translations.columns:
        if lang == "GEST_sentence" or lang == last_lang:
            continue
        translated_masc = list(gendered_translations[(lang, "the man said")])
        translated_fem = list(gendered_translations[(lang, "the woman said")])

        if len(translated_masc) == 0 or pd.isna(translated_masc[0]):
            continue  # skip empty columns

        # Score male
        sys_score_masc, sent_scores_masc = score_translations(sources_masc, translated_masc)
        idxs_to_remove_masc = highlight_bad_translations(sent_scores_masc, gest_sentences, translated_masc)
        gendered_translations.loc[gendered_translations.index[idxs_to_remove_masc], (lang, "the man said")] = ''

        # Score female
        sys_score_fem, sent_scores_fem = score_translations(sources_fem, translated_fem)
        idxs_to_remove_fem = highlight_bad_translations(sent_scores_fem, gest_sentences, translated_fem)
        gendered_translations.loc[gendered_translations.index[idxs_to_remove_fem], (lang, "the woman said")] = ''

        system_scores_masc[lang] = sys_score_masc
        system_scores_fem[lang] = sys_score_fem
        sentence_scores_masc[lang] = sent_scores_masc
        sentence_scores_fem[lang] = sent_scores_fem

        last_lang = lang

    # ---------- SAVE FILTERED DATASETS ----------
    neutral_translations.to_csv(FILTERED_DIR / f"neutral_stereotype_{stereotype_number}_filtered.csv")
    gendered_translations.to_csv(FILTERED_DIR / f"gendered_stereotype_{stereotype_number}_filtered.csv")

    # ---------- SAVE SYSTEM SCORES ----------
    average_gendered_scores = {lang: (system_scores_masc[lang] + system_scores_fem[lang])/2
                               for lang in system_scores_masc}

    system_scores_neutral_df = pd.DataFrame([system_scores_neutral])
    system_scores_gendered_df = pd.DataFrame([average_gendered_scores])

    system_scores_all = pd.concat([system_scores_neutral_df, system_scores_gendered_df], axis=1).T
    system_scores_all["Coverage"] = np.nan

    # Compute coverage (count missing translations)
    neutral_filtered = neutral_translations
    gendered_filtered = gendered_translations

    for lang in neutral_filtered.columns:
        if lang == "GEST_sentence":
            continue
        system_scores_all.loc[lang, "Coverage"] = neutral_filtered[lang].isna().sum()

    last_lang = None
    for (lang, condition) in gendered_filtered.columns:
        if lang == "GEST_sentence" or lang == last_lang:
            continue
        mask = gendered_filtered[(lang, "the man said")].isna() | gendered_filtered[(lang, "the woman said")].isna()
        system_scores_all.loc[lang, "Coverage"] = mask.sum()
        last_lang = lang

    system_scores_all.to_csv(FILTERED_DIR / f"qe_scores_{stereotype_number}.csv")
    print(f"Completed QE filtering for stereotype {stereotype_number}")