"""
Download Dutch and German splits from EuroGEST
"""

import argparse
from pathlib import Path

from datasets import load_dataset

LANGUAGES = ["German", "Dutch"]
HF_DATASET = "utter-project/EuroGEST"
OUTPUT_DIR = "./data/"


def download(languages=LANGUAGES, output_dir=OUTPUT_DIR, hf_token=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(HF_DATASET, token=hf_token)

    for lang in languages:
        df = dataset[lang].to_pandas()
        out_path = output_dir / f"{lang}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {lang}: {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    download(LANGUAGES, OUTPUT_DIR, args.hf_token)
