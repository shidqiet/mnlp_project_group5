"""
Download Dutch and German splits from EuroGEST
"""

import argparse
from pathlib import Path

from datasets import load_dataset

LANGUAGES = ["German", "Dutch"]
HF_DATASET = "utter-project/EuroGEST"
OUTPUT_DIR = "./data/"


def download(hf_token=None):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(HF_DATASET, token=hf_token)

    for lang in LANGUAGES:
        df = dataset[lang].to_pandas()
        out_path = output_dir / f"{lang}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {lang}: {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    download(args.hf_token)
