#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from pathlib import Path
import os
import numpy as np
import fire

# ============================================================= #
# ================== CONFIGURATION ============================= #
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
STEREOTYPES = range(1, 17)

# ===================== UTILITY FUNCTIONS ===================== #

def extract_quoted_sentence(text):
    ''' function to extract the stereotype sentence from quotation marks in the
    gender-forced translated sentence '''
    text = text.replace("‘", "'").replace("’", "'")
    quoted_sentence = re.search(r'«([^"]+)»', text)
    if quoted_sentence == None: 
        quoted_sentence = re.search(r'"([^"]+)"', text)
        if quoted_sentence == None:
            quoted_sentence = re.search(r"'([^']+)'", text)
            ## deal with exceptions where ' is used mid phrase
            if quoted_sentence and len(quoted_sentence[1]) <= 3:
                quoted_sentence = re.search(r"'(.*)'", text)
            if quoted_sentence == None:
                quoted_sentence = re.search(r'「([^"]+)」', text)
                if quoted_sentence == None: 
                    quoted_sentence = re.search(r'„([^"]+)(“|”)', text)
                    if quoted_sentence == None:
                        quoted_sentence = re.search(r'“([^”]+)', text)
                        if quoted_sentence == None:
                            quoted_sentence = re.search("‚([^’]+)", text)
                            if quoted_sentence == None:
                                quoted_sentence = re.search(",([^']+)", text)
                                if quoted_sentence == None:
                                    print("Error in extracting quoted sentence (missing opening or closing quote marks)")
                                    print(text) 
                                    quoted_sentence = [f'"ERROR": {text}']
    if quoted_sentence == None:
        print("Error in extracting quoted sentence")
        print(text) 
        quoted_sentence = [f'"ERROR": {text}']
        print(f"Error in extracting quoted sentence from: {text}", flush=True)
        return "ERROR"
    else:
        quoted_sentence = quoted_sentence[0].strip()
        # strip punc
        pattern = r'^[^\w\s]+|[^\w\s]+$'
        quoted_sentence = re.sub(pattern, '', quoted_sentence) 
        return quoted_sentence


def compare_translations(masculine_translation, feminine_translation, num_gendered_words, num_different_letters):
    gendered_words = []
    if pd.isna(masculine_translation) or pd.isna(feminine_translation):   
        return "Unknown", None
    else:
        # remove any punctuation remaining inside the sentence 
        pattern = r'[^\w\s]'
        masculine_translation = re.sub(pattern, '', masculine_translation, flags=re.UNICODE).strip() 
        feminine_translation = re.sub(pattern, '', feminine_translation, flags=re.UNICODE).strip() 
        pattern = r'  '
        masculine_translation = re.sub(pattern, ' ', masculine_translation, flags=re.UNICODE) 
        feminine_translation = re.sub(pattern, ' ', feminine_translation, flags=re.UNICODE) 
        if masculine_translation == feminine_translation:
            return "Neutral", None
        else:
            ## check if character length difference is greater than 5
            if abs(len(masculine_translation) - len(feminine_translation)) > 5:
                return "Unknown", None
            else:
                # check if transaltions differ by more than 2 words 
                masculine_words = masculine_translation.split()
                feminine_words = feminine_translation.split()
                if len(masculine_words) != len(feminine_words):
                    return "Unknown", None
                else:
                    count_different_words = 0
                    different_word_indexes = []
                    for word_index, (masculine_word, feminine_word) in enumerate(zip(masculine_words, feminine_words)):
                        if masculine_word != feminine_word:
                            count_different_words += 1
                            different_word_indexes.append(word_index)
                    if count_different_words > num_gendered_words:
                        return "Unknown", None
                    else:
                        if len(different_word_indexes) == 0:
                            print(masculine_translation)
                            print(feminine_translation)
                        else:
                            # check if different words differ by only two lertters 
                            for different_word_index in different_word_indexes:
                                masculine_word = masculine_words[different_word_index]
                                feminine_word = feminine_words[different_word_index]
                                masc_length = len(masculine_word)
                                fem_length = len(feminine_word)
                                longest_word_length = min(masc_length, fem_length)
                                num_different_letters_length = np.abs(masc_length-fem_length)
                                max_different_letters = num_different_letters - num_different_letters_length
                                
                                count_different_letters = 0
                                for letter_index in range(0, longest_word_length):
                                    if masculine_word[letter_index].casefold() == feminine_word[letter_index].casefold():
                                        pass
                                    else:
                                        count_different_letters += 1
                                if count_different_letters <= max_different_letters:
                                    gendered_words.append((masculine_word, feminine_word))
                                else:                                            
                                    return "Unknown", None
            if len(gendered_words) >= 1:
                return "Gendered", gendered_words
            

def count_non_null_rows(df, column):
    total = 0
    sentences = df[column]
    for row in sentences:
        if not pd.isna(row):
            total += 1
    return total


def count_stereotype_samples(language_data: pd.DataFrame, stereotypes: list) -> dict:
    """
    Filters a DataFrame to include only rows with at least one translation 
    (neutral, masculine, or feminine) and then counts the number of samples 
    for each stereotype based on the filtered data.
    """

    translation_cols = [
        "Neutral translation", 
        "Masculine translation"
    ]

    all_missing_mask = language_data[translation_cols].isna().all(axis=1)
    at_least_one_translation_mask = ~all_missing_mask
    translated_data = language_data[at_least_one_translation_mask].copy()

    num_samples_per_stereotype = {}
    
    for stereotype in STEREOTYPES:
        num_samples = len(translated_data[translated_data["original_stereotype"] == stereotype])
        num_samples_per_stereotype[stereotype] = num_samples
        
    return num_samples_per_stereotype



# ===================== MAIN FUNCTION ===================== #

def main(
    languages,
    gest_path,
    input_dir,
    out_dir,
    num_gendered_words,
    num_different_letters,
):
    """
    Apply heuristic filtering to separate neutral, gendered, and invalid translations.
    """
    gest_df = pd.read_csv(gest_path)

    in_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "discarded").mkdir(parents=True, exist_ok=True)

    # 4. Handle Language Selection
    if languages:
        if isinstance(languages, str):
            languages = [l.strip() for l in languages.split(',')]
        target_langs = {l: ALL_LANGS[l] for l in languages if l in ALL_LANGS}
    else:
        target_langs = ALL_LANGS
    
    stats_cols = ["Neutral",
                "Gendered",
                "Discarded Heuristic",
                "Discarded QE",
                "Coverage",
                "Number samples per stereotype"]
    stats_df = pd.DataFrame(index=target_langs, columns=stats_cols)

    for lang in target_langs:
        print(f"--- Finalizing {lang} ---")
        filtered_file = in_path / f"{lang.lower().replace(' ', '_')}_filtered.csv"
        
        if not filtered_file.exists():
            print(f"Skipping {lang}: Filtered file not found.")
            continue

        filtered_df = pd.read_csv(filtered_file, index_col=0)
        
        # Result structures
        lang_data = pd.DataFrame(columns=["GEST sentence", 
                                 "Neutral translation", 
                                 "Masculine translation", 
                                 "Feminine translation", 
                                 "Gendered word masculine",
                                 "Gendered word feminine",
                                 "Original_stereotype"])
        lang_data["GEST sentence"] = gest_df["sentence"]
        lang_data["original_stereotype"] = gest_df["stereotype"]

        if lang in NEUTRAL_LANGS:
            clean_df = filtered_df.dropna(subset=["translation"], how="any")
            stats_df.at[lang, "Discarded QE"] = len(gest_df) - len(clean_df)
            
            for idx, row in clean_df.iterrows():
                lang_data.at[idx, "Neutral translation"] = row["translation"]

        else: # Gendered Langs
            clean_df = filtered_df.dropna(subset=["the man said", "the woman said"], how='any')
            stats_df.at[lang, "Discarded QE"] = len(gest_df) - len(clean_df)

            unknown_data = pd.DataFrame(columns=["GEST sentence", "masculine translation", "feminine translation", "original_stereotype"])
            unknown_data["GEST sentence"] = gest_df["sentence"]
            unknown_data["original_stereotype"] = gest_df["stereotype"]

            for idx, row in clean_df.iterrows():
                m_sent = extract_quoted_sentence(row["the man said"])
                f_sent = extract_quoted_sentence(row["the woman said"])
                
                res, words = compare_translations(m_sent, f_sent, num_gendered_words, num_different_letters)
                
                if res == "Neutral":
                    lang_data.at[idx, "Neutral translation"] = m_sent
                elif res == "Gendered":
                    lang_data.at[idx, "Masculine translation"] = m_sent
                    lang_data.at[idx, "Feminine translation"] = f_sent
                    lang_data.at[idx, "Gendered word masculine"] = [w[0] for w in words]
                    lang_data.at[idx, "Gendered word feminine"] = [w[1] for w in words]
                else: # Unknown
                    unknown_data.at[idx, "Masculine translation"] = m_sent
                    unknown_data.at[idx, "Feminine translation"] = f_sent
                
                # Save and count Unknowns
                u_count = unknown_data.get("Masculine translation", pd.Series(dtype=float)).count()
                unknown_data.to_csv(out_path / "discarded" / f"{lang.lower()}_discarded.csv")
                # unknown_data.dropna(subset=["Masculine translation"]).to_csv(out_path / "discarded" / f"{lang.lower()}_discarded.csv")
        
        n_count = lang_data["Neutral translation"].count()
        g_count = lang_data.get("Masculine translation", pd.Series()).count()

        num_samples_per_stereotype = count_stereotype_samples(lang_data, STEREOTYPES)
        n = n_count if n_count is not None else 0
        g = g_count if g_count is not None else 0
        u = u_count if u_count is not None else 0

        stats_df.at[lang, "Neutral"], stats_df.at[lang, "Gendered"] = n, g
        stats_df.at[lang, "Discarded Heuristic"] = u
        stats_df.at[lang, "Coverage"] = ((n + g) / len(gest_df)) * 100
        stats_df.at[lang, "Number samples per stereotype"] = str(num_samples_per_stereotype)

        # Save Final CSV
        lang_data.to_csv(out_path / f"{lang.lower().replace(' ', '_')}_final.csv")

    stats_df.to_csv(out_path / f"stats_summary.csv")
    print(f"Heuristic filtering complete. Results in: {out_dir}")

if __name__ == "__main__":
    fire.Fire(main)