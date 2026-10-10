"""
Masculine rank of each stereotype per language and model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT = "results/all_sentence_scores.csv"
OUTPUT = "plots/"

STEREOTYPE_LABELS = {
    1: "Emotional",
    2: "Gentle",
    3: "Empathetic",
    4: "Neat",
    5: "Social",
    6: "Weak",
    7: "Beautiful",
    8: "Tough",
    9: "Self-conf.",
    10: "Professional",
    11: "Rational",
    12: "Providers",
    13: "Leaders",
    14: "Childish",
    15: "Sexual",
    16: "Strong",
}
LANG_ORDER = ["en", "id", "zh", "de", "nl"]
MAIN_LANGS = ["id", "zh", "de", "nl"]
LANG_NAMES = {
    "en": "English",
    "id": "Indonesian",
    "zh": "Chinese",
    "de": "German",
    "nl": "Dutch",
}
MODEL_SHORT = {
    "meta-llama/Llama-2-7b-hf": "Llama-2",
    "Yellow-AI-NLP/komodo-7b-base": "Komodo",
    "hfl/chinese-llama-2-7b": "Zh-LLaMA-2",
    "LeoLM/leo-hessianai-7b": "LeoLM",
    "ChocoLlama/ChocoLlama-2-7B-base": "ChocoLlama",
    "LLaMAX/LLaMAX2-7B": "LLaMAX",
}
MODEL_TYPES = [
    ("base", "Base (Llama-2)"),
    ("multi_adapted", "Multilingual-adapted (LLaMAX2)"),
    ("lang_adapted", "Language-adapted"),
]


def compute_qi(df):
    df = df.copy()
    df["stereotype_id"] = df["stereotype_id"].astype(int)
    per = (
        df.groupby(["language", "model", "model_type", "stereotype_id"])["masc_ratio"]
        .agg(q_i="mean", n="count")
        .reset_index()
    )
    per.loc[per["n"] < 10, "q_i"] = float("nan")
    return per


def compute_rankings(qi_df):
    rows = []
    for (lang, model, mtype), g in qi_df.groupby(["language", "model", "model_type"]):
        ranked = (
            g.dropna(subset=["q_i"])
            .sort_values("q_i", ascending=False)
            .reset_index(drop=True)
        )
        ranked["rank"] = ranked.index + 1
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def print_rankings(rank_df):
    for mtype, _ in MODEL_TYPES:
        print(f"\n=== {mtype.upper()} ===")
        sub = rank_df[rank_df["model_type"] == mtype]
        pivot = pd.DataFrame(index=range(1, 17))
        for lang in LANG_ORDER:
            lang_sub = sub[sub["language"] == lang]
            if lang_sub.empty:
                continue
            s = lang_sub.set_index("stereotype_id")["rank"]
            pivot[LANG_NAMES[lang]] = s
        pivot.index = [STEREOTYPE_LABELS[i] for i in pivot.index]
        print(pivot.to_string())


def plot_heatmap(rank_df, output_dir, model_types, langs, name):
    rows = [
        rank_df[(rank_df["model_type"] == m) & rank_df["language"].isin(langs)]["language"].nunique()
        for m, _ in model_types
    ]
    _, axes = plt.subplots(
        len(model_types), 1, figsize=(14, 1.5 + 0.7 * sum(rows)),
        sharex=True, gridspec_kw={"height_ratios": rows},
    )

    for (mtype, title), ax in zip(model_types, axes):
        sub = rank_df[rank_df["model_type"] == mtype]
        pivot = pd.DataFrame(columns=range(1, 17))
        for lang in langs:
            lang_sub = sub[sub["language"] == lang]
            if not lang_sub.empty:
                pivot.loc[LANG_NAMES[lang]] = lang_sub.set_index("stereotype_id")["rank"]
        pivot = pivot.dropna(how="all").astype(float)
        pivot.columns = [STEREOTYPE_LABELS[i] for i in pivot.columns]

        sns.heatmap(
            pivot, ax=ax, cmap="PuOr_r", center=8.5, vmin=1, vmax=16,
            annot=True, fmt=".0f",
            cbar_kws={"label": "Rank (1 = most masculine)"},
        )
        ax.axvline(7, color="red", linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Language", fontsize=10)
        ax.set_xlabel("")

    axes[-1].set_xlabel("Stereotype", fontsize=10)
    axes[-1].tick_params(axis="x", rotation=45)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    plt.suptitle("Masculine rank per stereotype per language", fontsize=13)
    plt.tight_layout()

    out = output_dir / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    df = pd.read_csv(INPUT)
    df["model"] = df["model"].map(lambda m: MODEL_SHORT.get(m, m))

    qi_df = compute_qi(df)
    rank_df = compute_rankings(qi_df)

    print_rankings(rank_df)

    out_dir = Path(OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(rank_df, out_dir, MODEL_TYPES, LANG_ORDER, "stereotype_ranking_heatmap.png")
    plot_heatmap(rank_df, out_dir, MODEL_TYPES[1:], MAIN_LANGS, "stereotype_ranking_heatmap_main.png")


if __name__ == "__main__":
    main()
