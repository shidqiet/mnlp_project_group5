"""
Print g_s summary table from all_sentence_scores.csv
"""

import pandas as pd

INPUT = "results/all_sentence_scores.csv"

FEM_IDS = list(range(1, 8))
MASC_IDS = list(range(8, 17))

MODEL_SHORT = {
    "meta-llama/Llama-2-7b-hf": "Llama-2",
    "Yellow-AI-NLP/komodo-7b-base": "Komodo",
    "hfl/chinese-llama-2-7b": "Zh-LLaMA-2",
    "LeoLM/leo-hessianai-7b": "LeoLM",
    "ChocoLlama/ChocoLlama-2-7B-base": "ChocoLlama",
    "LLaMAX/LLaMAX2-7B": "LLaMAX",
}

df = pd.read_csv(INPUT)
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
            "q_f": round(q_f, 4),
            "q_m": round(q_m, 4),
            "g_s": round(q_m / q_f, 4) if q_f > 0 else None,
        }
    )

out = pd.DataFrame(rows).sort_values(["language", "model_type"])
print(out.to_string(index=False))
