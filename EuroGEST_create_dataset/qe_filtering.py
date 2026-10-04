#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
from pathlib import Path
import pandas as pd
from comet import download_model, load_from_checkpoint
import torch
import numpy as np
import fire

# ============================================================= #
# ================== CONFIGURATION ============================ #
# ============================================================= #

GENDERED_LANGS = {
    'Bulgarian': 'bg', 'French': 'fr', 'German': 'de', 'Greek': 'el',
    'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt',
    'Portuguese': 'pt', 'Romanian': 'ro', 'Spanish': 'es', 'Catalan': 'ca',
    'Galician': 'gl', 'Croatian': 'hr', 'Czech': 'cs', 'Polish': 'pl',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Russian': 'ru', 'Ukrainian': 'uk'
}

NEUTRAL_LANGS = {
    'Danish': 'da', 'Dutch': 'nl', 'Estonian': 'et', 'Finnish': 'fi',
    'Hungarian': 'hu', 'Irish': 'ga', 'Swedish': 'sv',
    'Norwegian': 'no', 'Turkish': 'tr'
}

ALL_LANGS = {**GENDERED_LANGS, **NEUTRAL_LANGS}

# ===================== UTILITY FUNCTIONS ===================== #

def score_translations(sources, targets, model, batch_size=16, gpus=1):
    samples = [{"src": s, "mt": t} for s, t in zip(sources, targets)]
    outputs = model.predict(
        samples=samples,
        batch_size=batch_size,
        gpus=gpus,
        accelerator="auto"
    )
    return outputs.system_score, outputs.scores

def highlight_bad_translations(sentence_scores, sources, targets, threshold):
    indexes_to_remove = []
    for i, score in enumerate(sentence_scores):
        if score <= threshold:
            print(f"Bad translation (score={score:.2f}):")
            print(f"Source: {sources[i]} | Target: {targets[i]}")
            indexes_to_remove.append(i)
    return indexes_to_remove

# ===================== MAIN FUNCTION ===================== #

def main(
    languages, 
    threshold, 
    batch_size,
    qe_model_name,
    data_dir,
    out_dir
):
    """
    Filter translations using COMET-QE.
    
    :param languages: List of languages or comma-separated string (e.g. "Spanish,French"). If None, runs all.
    :param threshold: QE score below which translations are removed.
    :param batch_size: Inference batch size.
    :param qe_model_name: The COMET model to use.
    :param data_dir: Path to raw CSV files.
    :param out_dir: Path to save filtered CSV files.
    """
    
    # 1. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Path Handling
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 3. Model Loading
    print(f"Loading QE Model: {qe_model_name}")
    model_path = download_model(qe_model_name)
    qe_model = load_from_checkpoint(model_path)
    qe_model.to(device)
    qe_model.eval()

    # 4. Handle Language Selection
    if languages:
        if isinstance(languages, str):
            languages = [l.strip() for l in languages.split(',')]
        target_langs = {l: ALL_LANGS[l] for l in languages if l in ALL_LANGS}
    else:
        target_langs = ALL_LANGS

    # Stats tracking
    system_scores_neutral = {}
    system_scores_gendered = {}
    coverage = {}

    for language in target_langs.keys():
        print(f"\n--- Processing {language} ---")
        raw_file = data_path / f"{language.lower().replace(' ', '_')}.csv"

        if not raw_file.exists():
            print(f"File {raw_file} not found. Skipping.")
            continue

        raw_df = pd.read_csv(raw_file, index_col=0)
        idxs_to_remove = set()

        if language in NEUTRAL_LANGS:
            srcs = list(raw_df["GEST_sentence"])
            tgts = list(raw_df["translation"])
            sys_s, sent_s = score_translations(srcs, tgts, qe_model, batch_size)
            
            idxs_to_remove = highlight_bad_translations(sent_s, srcs, tgts, threshold)
            system_scores_neutral[language] = sys_s

        elif language in GENDERED_LANGS:
            orig = list(raw_df["GEST_sentence"])
            src_m = [f"The man said, '{s}'" for s in orig]
            src_f = [f"The woman said, '{s}'" for s in orig]
            tgt_m = list(raw_df["the man said"])
            tgt_f = list(raw_df["the woman said"])

            # Score Masc
            sys_m, sent_m = score_translations(src_m, tgt_m, qe_model, batch_size)
            idxs_m = highlight_bad_translations(sent_m, src_m, tgt_m, threshold)
            
            # Score Fem
            sys_f, sent_f = score_translations(src_f, tgt_f, qe_model, batch_size)
            idxs_f = highlight_bad_translations(sent_f, src_f, tgt_f, threshold)

            idxs_to_remove = set(idxs_m).union(set(idxs_f))
            system_scores_gendered[language] = (sys_m, sys_f, (sys_m + sys_f) / 2)

        # Filtering
        if len(idxs_to_remove) > 0:
            idxs_to_drop = raw_df.index[list(idxs_to_remove)]
            filtered_df = raw_df.drop(idxs_to_drop)
        else:
            filtered_df = raw_df

        # Save
        filtered_df.to_csv(out_path / f"{language.lower().replace(' ', '_')}_filtered.csv")
        coverage[language] = len(filtered_df) / len(raw_df)
        print(f"Language {language} done. Coverage: {coverage[language]:.2%}")

    # 5. Final Aggregation
    if system_scores_neutral or system_scores_gendered:
        neutral_df = pd.Series(system_scores_neutral, name="Avg All Score").to_frame()
        gendered_df = pd.DataFrame(system_scores_gendered).T
        if not gendered_df.empty:
            gendered_df.columns = ["Avg Masc Score", "Avg Fem Score", "Avg All Score"]
        
        final_scores = pd.concat([neutral_df, gendered_df], axis=1)
        # Fix column alignment if one set is empty
        if "Avg All Score" in gendered_df.columns and "Avg All Score" in neutral_df.columns:
            unified = gendered_df["Avg All Score"].combine_first(neutral_df["Avg All Score"])
            final_scores = final_scores.drop(columns=["Avg All Score"])
            final_scores["Avg All Score"] = unified

        final_scores.to_csv(out_path / "system_scores.csv")
        pd.Series(coverage, name="Coverage").to_csv(out_path / "coverage.csv")

if __name__ == "__main__":
    fire.Fire(main)