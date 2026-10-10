#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add COMET-QE scores to the translated GEST files.

Reads the CSVs written by the translation scripts (row index = GEST row, GEST_sentence, translation,
original_stereotype) and writes the same rows with a COMET_score column added.

Unlike qe_filtering.py, which drops every row at or below the threshold and keeps no scores, nothing
is dropped here: each row keeps its score, so the threshold can be chosen - and changed - afterwards.
The score is what EuroGEST scores for gender-neutral languages: the reference-free COMET-QE score of
(English GEST sentence, translation).
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import fire
import pandas as pd
import torch

# ============================================================= #
# ================== CONFIGURATION ============================ #
# ============================================================= #

GENDERED_LANGS = {
    'Bulgarian': 'BG', 'French': 'FR', 'German': 'DE', 'Greek': 'EL',
    'Italian': 'IT', 'Latvian': 'LV', 'Lithuanian': 'LT', 'Maltese': 'MT',
    'Portuguese': 'PT', 'Romanian': 'RO', 'Spanish': 'ES', 'Catalan': 'CA',
    'Galician': 'GL', 'Croatian': 'HR', 'Czech': 'CS', 'Polish': 'PL',
    'Slovak': 'SK', 'Slovenian': 'SL', 'Russian': 'RU', 'Ukrainian': 'UK'
}

NEUTRAL_LANGS = {
    'Danish': 'DA', 'Dutch': 'NL', 'Estonian': 'ET', 'Finnish': 'FI',
    'Hungarian': 'HU', 'Irish': 'GA', 'Swedish': 'SV',
    'Norwegian': 'NB', 'Turkish': 'TR', 'Chinese': 'ZH', 'Indonesian': 'ID'
}

ALL_LANGS = {**GENDERED_LANGS, **NEUTRAL_LANGS}

# The checkpoint is pinned by revision so the numbers stay reproducible.
QE_MODEL = "Unbabel/wmt22-cometkiwi-da"
QE_MODEL_REVISION = "1ad785194e391eebc6c53e2d0776cada8f83179a"

SOURCE_COLUMN = "GEST_sentence"
TRANSLATION_COLUMN = "translation"
STEREOTYPE_COLUMN = "original_stereotype"
SCORE_COLUMN = "COMET_score"
GENDERED_COLUMNS = ("the man said", "the woman said")

# ===================== UTILITY FUNCTIONS ===================== #


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qe_model(qe_model_name, revision):
    """Download the pinned checkpoint and load it."""
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    # max_workers=1: with parallel workers, huggingface_hub's symlink probe races on Windows without
    # Developer Mode (threads call os.symlink before the probe returns False) -> WinError 1314.
    path = snapshot_download(qe_model_name, revision=revision, max_workers=1)
    model = load_from_checkpoint(str(Path(path) / "checkpoints" / "model.ckpt"))
    model.eval()
    return model


def check_against_gest(df, gest, language):
    """Fail loudly if the file's rows do not line up with gest_1.1.csv.

    The row index is the GEST row number, so a shifted or re-sorted file would otherwise be scored
    against the wrong English sentences without anyone noticing.
    """
    problems = []
    ids = pd.to_numeric(pd.Series(df.index), errors="coerce")
    if ids.isna().any() or (ids % 1 != 0).any():
        problems.append("the first column must hold GEST row numbers")
    elif not ids.between(0, len(gest) - 1).all():
        problems.append(f"GEST row numbers outside 0..{len(gest) - 1}")
    elif ids.duplicated().any():
        problems.append(f"duplicate GEST row numbers: {sorted(ids[ids.duplicated()].astype(int))[:10]}")
    else:
        rows = gest.loc[ids.astype(int).to_numpy()]
        mismatched = df[SOURCE_COLUMN].to_numpy() != rows["sentence"].to_numpy()
        if mismatched.any():
            examples = [f"row {i}: {s!r} != GEST {g!r}" for i, s, g in
                        zip(ids[mismatched][:3], df[SOURCE_COLUMN].to_numpy()[mismatched][:3], rows["sentence"].to_numpy()[mismatched][:3])]
            problems.append(f"{mismatched.sum()} rows disagree with GEST: " + "; ".join(examples))
        if STEREOTYPE_COLUMN in df.columns:
            wrong = pd.to_numeric(df[STEREOTYPE_COLUMN], errors="coerce").to_numpy() != rows["stereotype"].to_numpy()
            if wrong.any():
                problems.append(f"{wrong.sum()} rows have a stereotype that differs from GEST")
    if problems:
        raise ValueError(f"{language}: " + "; ".join(problems))


class ScoreCache:
    """Scores already computed, keyed by (model, source, translation).

    Scoring is the slow part, so a rerun after adding a language - or after an interrupted run -
    costs nothing and returns exactly the same numbers.
    """

    def __init__(self, path):
        self.conn = None
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, model TEXT, src TEXT, mt TEXT,"
            " score REAL NOT NULL, created_at TEXT)"
        )

    @staticmethod
    def key(model, src, mt):
        return hashlib.sha256("\x1f".join((model, src, mt)).encode("utf-8")).hexdigest()

    def get(self, keys):
        if self.conn is None:
            return {}
        found = {}
        keys = list(dict.fromkeys(keys))
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            for key, score in self.conn.execute(f"SELECT key, score FROM scores WHERE key IN ({placeholders})", chunk):
                found[key] = score
        return found

    def put(self, rows):
        if self.conn is None:
            return
        with self.conn:
            self.conn.executemany("INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?, ?, ?)", rows)

    def close(self):
        if self.conn is not None:
            self.conn.close()


def score_pairs(pairs, load_model, model_name, cache, batch_size, gpus):
    """COMET-QE scores for (source, translation) pairs; only pairs new to the cache reach the model.

    `load_model` is called only if something actually has to be scored, so a rerun that is fully
    cached never waits for the checkpoint to load.
    """
    keys = [ScoreCache.key(model_name, src, mt) for src, mt in pairs]
    known = cache.get(keys)
    todo = {}
    for key, pair in zip(keys, pairs):
        if key not in known and key not in todo:
            todo[key] = pair

    if todo:
        model = load_model()
        samples = [{"src": src, "mt": mt} for src, mt in todo.values()]
        outputs = model.predict(samples, batch_size=batch_size, gpus=gpus, accelerator="auto" if gpus else "cpu")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cache.put([(key, model_name, src, mt, float(score), now)
                   for (key, (src, mt)), score in zip(todo.items(), outputs.scores)])
        known = cache.get(keys)

    return [known[key] for key in keys], len(todo)


# ===================== MAIN FUNCTION ===================== #


def main(
    languages,
    data_dir,
    out_dir,
    gest_path="./gest_1.1.csv",
    batch_size=16,
    threshold=0.85,
    qe_model_name=QE_MODEL,
    qe_model_revision=QE_MODEL_REVISION,
    cache_path=None,
):
    """
    Add a COMET-QE score to every translated sentence.

    :param languages: Language or comma-separated list, matching the translated file names (e.g. "Chinese,Indonesian").
    :param data_dir: Folder with the translated CSVs (e.g. ./translated_output).
    :param out_dir: Folder for the scored CSVs.
    :param gest_path: gest_1.1.csv, used to check that the rows still line up with GEST.
    :param batch_size: Inference batch size.
    :param threshold: Only reported, never applied: no rows are dropped.
    :param qe_model_name: COMET model.
    :param qe_model_revision: Pinned revision of that model.
    :param cache_path: Score cache (default: <out_dir>/.qe_cache.sqlite; pass "" to disable).
    """
    data_path, out_path = Path(data_dir), Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(languages, str):
        languages = [lang.strip() for lang in languages.split(",") if lang.strip()]
    languages = [lang for lang in languages if lang]

    gest = pd.read_csv(gest_path, keep_default_na=False)
    device = pick_device()
    print(f"Using device: {device}")

    cache = ScoreCache(str(out_path / ".qe_cache.sqlite") if cache_path is None else cache_path)
    summary = []

    loaded = {}

    def load_model():
        if "model" not in loaded:
            print(f"Loading QE model: {qe_model_name} @ {qe_model_revision[:7]}")
            loaded["model"] = load_qe_model(qe_model_name, qe_model_revision)
        return loaded["model"]

    try:
        for language in languages:
            if language not in ALL_LANGS:
                print(f"Warning: {language} not in the supported list. Continuing anyway.")
            raw_file = data_path / f"{language.lower().replace(' ', '_')}.csv"
            if not raw_file.exists():
                print(f"File {raw_file} not found. Skipping.")
                continue

            df = pd.read_csv(raw_file, index_col=0, keep_default_na=False)
            missing = [c for c in (SOURCE_COLUMN, TRANSLATION_COLUMN) if c not in df.columns]
            if missing:
                raise ValueError(f"{raw_file} has no {missing} column; this script scores gender-neutral "
                                 "translations (one translation per sentence)")
            gendered = [c for c in GENDERED_COLUMNS if c in df.columns]
            if gendered:
                raise ValueError(f"{raw_file} holds gendered translations ({gendered}); score those with "
                                 "qe_filtering.py, or extend this script with the M/F templates")

            check_against_gest(df, gest, language)

            translated = df[TRANSLATION_COLUMN].astype(str).str.strip() != ""
            pairs = list(zip(df.loc[translated, SOURCE_COLUMN], df.loc[translated, TRANSLATION_COLUMN]))
            print(f"\n--- {language}: {len(df)} rows, {len(pairs)} translated ---")
            if len(pairs) < len(df):
                print(f"{len(df) - len(pairs)} rows have no translation and are left without a score")

            scores, computed = score_pairs(pairs, load_model, f"{qe_model_name}@{qe_model_revision}", cache,
                                           batch_size, 1 if device in ("cuda", "mps") else 0)
            print(f"{computed} scored now, {len(pairs) - computed} from the cache")

            df[SCORE_COLUMN] = pd.Series(scores, index=df.index[translated])
            save_file = out_path / raw_file.name
            df.to_csv(save_file)

            kept = (df[SCORE_COLUMN] > threshold).sum()
            print(f"{language}: min {df[SCORE_COLUMN].min():.4f}, median {df[SCORE_COLUMN].median():.4f}, "
                  f"max {df[SCORE_COLUMN].max():.4f}; {kept} of {len(df)} above {threshold} "
                  f"({kept / len(df):.1%}) -> {save_file}")
            summary.append({
                "language": language, "rows": len(df), "scored": int(df[SCORE_COLUMN].notna().sum()),
                "min": df[SCORE_COLUMN].min(), "median": df[SCORE_COLUMN].median(),
                "mean": df[SCORE_COLUMN].mean(), "max": df[SCORE_COLUMN].max(),
                "above_threshold": int(kept), "threshold": threshold,
                "model": f"{qe_model_name}@{qe_model_revision}", "device": device,
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
    finally:
        cache.close()

    if summary:
        summary_file = out_path / "qe_summary.csv"
        pd.DataFrame(summary).to_csv(summary_file, index=False)
        print(f"\nSummary -> {summary_file}")


if __name__ == "__main__":
    fire.Fire(main)
