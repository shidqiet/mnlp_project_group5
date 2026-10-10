"""
Rescore the Llama-2 sentence pairs with LLaMAX2.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_models.dutch_german.main import score_pair, setup

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
MODEL_ID = "LLaMAX/LLaMAX2-7B"
MODEL_TYPE = "multi_adapted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = setup()
    print(f"Device: {device}")

    df = pd.read_csv(args.input)
    df = df[df["model"] == BASE_MODEL].reset_index(drop=True)
    print(f"{len(df)} pairs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device).eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        scores = score_pair(row["masc_sentence"], row["fem_sentence"], tokenizer, model, device)
        rows.append({**row.to_dict(), **scores, "model": MODEL_ID, "model_type": MODEL_TYPE})

    out = pd.DataFrame(rows)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
