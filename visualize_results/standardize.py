"""
Merge all sentence-level results into one CSV.
"""

from pathlib import Path

import pandas as pd

ZH_ID_EN_ROOT = Path("/home/scur0713/experiments/h100-complete-20261003/results/full")
DE_NL_ROOT = Path("/home/scur0713/experiments/results")
LLAMAX = DE_NL_ROOT / "llamax.csv"
OUT = Path("results/all_sentence_scores.csv")

# model_type: base = Llama-2, multi_adapted = LLaMAX2, lang_adapted = language-adapted
ZH_ID_EN_SOURCES = [
    ("komodo/id/indonesian_comet_full_csv", "id", "lang_adapted"),
    ("llama2/id/indonesian_comet_full_csv", "id", "base"),
    ("chinese_llama2/zh/chinese_comet_full_csv", "zh", "lang_adapted"),
    ("llama2/zh/chinese_comet_full_csv", "zh", "base"),
    ("llama2/en/results", "en", "base"),
]
DE_NL_SOURCES = [
    ("german_leo.csv", "de", "lang_adapted"),
    ("german_llama2.csv", "de", "base"),
    ("dutch_choco.csv", "nl", "lang_adapted"),
    ("dutch_llama2.csv", "nl", "base"),
]

# gendered mistranslations, e.g. id "tampan", zh 帅/英俊 vs 漂亮/美
EXCLUDE = {
    "id": [334, 652, 910, 1572, 1602, 1992, 2135, 2169, 2409, 2544, 3461, 3472, 3475, 992, 1761],
    "zh": [3475, 1992, 327, 1602, 652, 3472, 3132, 3461, 1961, 3515, 3462, 3315, 637, 2163, 3254,
           3558, 3473, 796, 2408, 3299, 2146],
}

COLS = [
    "GEST_ID",
    "language",
    "stereotype_id",
    "stereotype",
    "source_sentence",
    "masc_sentence",
    "fem_sentence",
    "avg_ll_masc",
    "avg_ll_fem",
    "masc_ratio",
    "model",
    "model_type",
]


def load_zh_id_en(folder_key, language, model_type):
    df = pd.read_csv(ZH_ID_EN_ROOT / folder_key / "sentence_scores.csv")
    df = df.rename(
        columns={
            "sample_id": "GEST_ID",
            "masculine_sentence": "masc_sentence",
            "feminine_sentence": "fem_sentence",
            "gender_preference_score": "masc_ratio",
            "masculine_normalized_log_likelihood": "avg_ll_masc",
            "feminine_normalized_log_likelihood": "avg_ll_fem",
        }
    )
    df["language"] = language
    df["model_type"] = model_type
    df["stereotype_id"] = df["stereotype_id"].astype(int)
    return df[COLS]


def load_de_nl(csv_name, language, model_type):
    df = pd.read_csv(DE_NL_ROOT / csv_name)
    df = df.rename(
        columns={
            "Stereotype_ID": "stereotype_id",
            "Source": "source_sentence",
            "masc_text": "masc_sentence",
            "fem_text": "fem_sentence",
        }
    )
    df["language"] = language
    df["model_type"] = model_type
    df["stereotype_id"] = df["stereotype_id"].astype(int)
    df["stereotype"] = df["stereotype_id"].astype(str)
    return df[COLS]


def main():
    frames = [load_zh_id_en(*s) for s in ZH_ID_EN_SOURCES]
    frames += [load_de_nl(*s) for s in DE_NL_SOURCES]
    frames.append(pd.read_csv(LLAMAX)[COLS].assign(model_type="multi_adapted"))

    df = pd.concat(frames, ignore_index=True)
    for lang, ids in EXCLUDE.items():
        df = df[~((df["language"] == lang) & df["GEST_ID"].isin(ids))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"{len(df)} rows to {OUT}")
    print(df.groupby(["language", "model_type"]).size().to_string())


if __name__ == "__main__":
    main()
