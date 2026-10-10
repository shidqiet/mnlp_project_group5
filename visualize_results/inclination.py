"""
Inclination: deviation of each stereotype's q_i from the model's default masculine rate.
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


def compute_inclination(qi_df):
    rows = []
    for (lang, model, mtype), g in qi_df.groupby(["language", "model", "model_type"]):
        default_masc = g["q_i"].mean()  # proxy neutral baseline
        g = g.copy()
        g["inclination"] = g["q_i"] - default_masc
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def print_inclination(incl_df):
    for mtype, _ in MODEL_TYPES:
        print(f"\n=== {mtype.upper()} ===")
        sub = incl_df[incl_df["model_type"] == mtype]
        pivot = pd.DataFrame(index=range(1, 17))
        for lang in LANG_ORDER:
            lang_sub = sub[sub["language"] == lang]
            if lang_sub.empty:
                continue
            pivot[LANG_NAMES[lang]] = lang_sub.set_index("stereotype_id")[
                "inclination"
            ].round(4)
        pivot.index = [STEREOTYPE_LABELS[i] for i in pivot.index]
        print(pivot.to_string())


def plot_heatmap(incl_df, output_dir, model_types, langs, name):
    rows = [
        incl_df[(incl_df["model_type"] == m) & incl_df["language"].isin(langs)]["language"].nunique()
        for m, _ in model_types
    ]
    _, axes = plt.subplots(
        len(model_types), 1, figsize=(14, 1.5 + 0.7 * sum(rows)),
        sharex=True, gridspec_kw={"height_ratios": rows},
    )

    for (mtype, title), ax in zip(model_types, axes):
        sub = incl_df[incl_df["model_type"] == mtype]
        pivot = pd.DataFrame(columns=range(1, 17))
        for lang in langs:
            lang_sub = sub[sub["language"] == lang]
            if not lang_sub.empty:
                pivot.loc[LANG_NAMES[lang]] = lang_sub.set_index("stereotype_id")["inclination"]
        pivot = pivot.dropna(how="all").astype(float)
        pivot.columns = [STEREOTYPE_LABELS[i] for i in pivot.columns]

        sns.heatmap(
            pivot, ax=ax, cmap="bwr", center=0, vmin=-0.3, vmax=0.3,
            annot=True, fmt=".2f",
            cbar_kws={"label": "Inclination (q_i - default)"},
            annot_kws={"fontsize": 7},
        )
        ax.axvline(7, color="black", linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Language", fontsize=10)
        ax.set_xlabel("")

    axes[-1].set_xlabel("Stereotype", fontsize=10)
    axes[-1].tick_params(axis="x", rotation=45)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    plt.suptitle("Inclination: q_i deviation from default masculine rate", fontsize=13)
    plt.tight_layout()

    out = output_dir / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    df = pd.read_csv(INPUT)
    df["model"] = df["model"].map(lambda m: MODEL_SHORT.get(m, m))

    qi_df = compute_qi(df)
    incl_df = compute_inclination(qi_df)

    print_inclination(incl_df)

    out_dir = Path(OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(incl_df, out_dir, MODEL_TYPES, LANG_ORDER, "inclination_heatmap.png")
    plot_heatmap(incl_df, out_dir, MODEL_TYPES[1:], MAIN_LANGS, "inclination_heatmap_main.png")


if __name__ == "__main__":
    main()
