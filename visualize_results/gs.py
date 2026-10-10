"""
g_s bar chart: stereotype rate per language, Llama-2 vs LLaMAX2 vs language-adapted.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = "results/all_sentence_scores.csv"
OUTPUT = "plots/"

FEM_IDS = list(range(1, 8))
MASC_IDS = list(range(8, 17))

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
    ("base", "Base (Llama-2)", "#4C72B0"),
    ("multi_adapted", "Multilingual-adapted (LLaMAX2)", "#55A868"),
    ("lang_adapted", "Language-adapted", "#DD8452"),
]


def compute_gs(df):
    df = df.copy()
    df["stereotype_id"] = df["stereotype_id"].astype(int)
    rows = []
    for (lang, model, mtype), g in df.groupby(["language", "model", "model_type"]):
        per = (
            g.groupby("stereotype_id")["masc_ratio"]
            .agg(q_i="mean", n="count")
            .reset_index()
        )
        per = per[per["n"] >= 10]
        q_f = per[per["stereotype_id"].isin(FEM_IDS)]["q_i"].mean()
        q_m = per[per["stereotype_id"].isin(MASC_IDS)]["q_i"].mean()
        rows.append(
            {
                "language": lang,
                "model": MODEL_SHORT.get(model, model),
                "model_type": mtype,
                "q_f": q_f,
                "q_m": q_m,
                "g_s": q_m / q_f if q_f > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_gs(gs_df, output_dir, langs, name):
    x = np.arange(len(langs))
    width = 0.27

    _, ax = plt.subplots(figsize=(11, 5))
    bars = []
    for i, (mtype, label, color) in enumerate(MODEL_TYPES):
        vals = []
        for lang in langs:
            v = gs_df[(gs_df["language"] == lang) & (gs_df["model_type"] == mtype)]["g_s"].values
            vals.append(v[0] if len(v) else float("nan"))
        bars += ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.85)

    ax.axhline(
        1.0, color="red", linestyle="dotted", linewidth=1.5, label="g_s = 1.0 (no stereotyping)"
    )
    for b in x[:-1] + 0.5:
        ax.axvline(b, color="grey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs], fontsize=12)
    ax.set_ylabel("Overall stereotype rate (g_s)", fontsize=12)
    ax.set_title("Overall stereotype rate per language and model", fontsize=13)
    ax.set_ylim(0.9, 1.55)
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    out = output_dir / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    df = pd.read_csv(INPUT)
    gs_df = compute_gs(df)

    print(
        gs_df[["language", "model", "model_type", "q_f", "q_m", "g_s"]]
        .sort_values(["language", "model_type"])
        .to_string(index=False)
    )

    out_dir = Path(OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_gs(gs_df, out_dir, LANG_ORDER, "gs_bar_chart.png")
    plot_gs(gs_df, out_dir, MAIN_LANGS, "gs_bar_chart_main.png")


if __name__ == "__main__":
    main()
