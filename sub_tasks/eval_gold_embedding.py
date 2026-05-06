"""
Evaluate the Gemini Embedding API (gemini-embedding-001) on the gold data.

Pipeline:
  1. Load gold pairs (SentenceA, SentenceB, gold score)
  2. Embed all unique sentences via the Gemini Embedding API
  3. Compute cosine similarity for each pair
  4. Correlate cosine similarities vs gold scores → Pearson / Spearman

Usage:
    python -m sub_tasks.eval_gold_embedding \\
        --config supported_models_config/gemini_embedding_001_config.yaml

Optional CLI overrides:
    --gold-data  <path>   gold .xlsx to evaluate against
                          (default: data/all_gold_pairs_1000_scored.xlsx)
    --output-dir <path>   directory for result files
                          (default: results/eval_gold_embedding/<timestamp>/)
    --use-unicode         use SentenceA/B_unicode columns instead of EWTS
    --resume              skip embedding if output CSV already exists
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_config import AppConfig
from common_utils import load_yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_GOLD_FILENAME = "all_gold_pairs_1000_scored.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_api_key(keys_path: str, env_var: str) -> str:
    keys = load_yaml(keys_path, must_exist=True)
    if env_var not in keys:
        raise KeyError(f"'{env_var}' not found in keys file: {keys_path}")
    return str(keys[env_var])


def _embed_sentences(
    sentences: List[str],
    model: str,
    api_key: str,
    task_type: str,
    batch_size: int,
    requests_per_minute: int,
) -> Dict[str, List[float]]:
    """Embed a list of sentences in batches, returning a dict sentence → vector."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    embeddings: Dict[str, List[float]] = {}
    unique = list(dict.fromkeys(sentences))  # preserve order, deduplicate
    n = len(unique)
    min_delay = 60.0 / requests_per_minute  # seconds between requests

    for start in range(0, n, batch_size):
        batch = unique[start: start + batch_size]
        t0 = time.monotonic()
        result = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        for text, emb in zip(batch, result.embeddings):
            embeddings[text] = emb.values

        done = start + len(batch)
        LOGGER.info("Embedded %d / %d sentences", done, n)

        # simple rate-limit: sleep if the request was faster than min_delay
        elapsed = time.monotonic() - t0
        if elapsed < min_delay and done < n:
            time.sleep(min_delay - elapsed)

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def eval_gold_embedding(
    config_path: str,
    gold_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_unicode: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """Evaluate Gemini embeddings against the gold similarity data.

    Args:
        config_path: Path to the embedding model YAML config.
        gold_data_path: Override path to the gold .xlsx file.
        output_dir: Directory for result files.
        use_unicode: Use SentenceA/B_unicode columns instead of EWTS.
        resume: Skip embedding step if the scores CSV already exists.

    Returns:
        Single-row DataFrame with Pearson/Spearman metrics.
    """
    app_config = AppConfig()
    cfg = load_yaml(config_path, must_exist=True)

    model: str = cfg.get("model", "")
    if not model:
        raise ValueError(f"'model' key missing from config: {config_path}")

    api_key_env: str = cfg.get("api_key_env", "GOOGLE_API_KEY")
    task_type: str = cfg.get("task_type", "SEMANTIC_SIMILARITY")
    batch_size: int = int(cfg.get("batch_size", 100))
    requests_per_minute: int = int(cfg.get("requests_per_minute", 1500))

    # ── paths ─────────────────────────────────────────────────────────────────
    _fallback_base = str(Path(__file__).resolve().parent.parent)
    sensim_base = str(Path(app_config.get("sensim_base_dir", _fallback_base)).resolve())

    if gold_data_path is None:
        gold_data_path = os.path.join(sensim_base, "data", DEFAULT_GOLD_FILENAME)
    if not Path(gold_data_path).exists():
        raise FileNotFoundError(f"Gold data file not found: {gold_data_path}")

    if output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(sensim_base, "results", "eval_gold_embedding", ts)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    scores_csv = output_dir / "embedding_scores.csv"

    # ── load gold data ────────────────────────────────────────────────────────
    print(f"Loading gold data: {gold_data_path}")
    gold_df = pd.read_excel(gold_data_path, engine="openpyxl")

    col_a = "SentenceA_unicode" if use_unicode else "SentenceA"
    col_b = "SentenceB_unicode" if use_unicode else "SentenceB"
    for col in (col_a, col_b, "ID", "score"):
        if col not in gold_df.columns:
            raise ValueError(f"Gold data missing column '{col}'")

    gold_df["ID"] = gold_df["ID"].astype(str)
    print(f"  {len(gold_df)} pairs | text: {'unicode' if use_unicode else 'EWTS'} | model: {model}")

    # ── embed & score (or resume) ─────────────────────────────────────────────
    if resume and scores_csv.exists():
        print(f"Resuming: found existing scores ({scores_csv.name})")
        scores_df = pd.read_csv(scores_csv, dtype={"ID": str})
    else:
        keys_path = str((Path(sensim_base) / cfg.get("keys_path", "keys.yaml")).resolve())
        api_key = _load_api_key(keys_path, api_key_env)

        all_sentences = gold_df[col_a].tolist() + gold_df[col_b].tolist()
        print(f"Embedding {len(set(all_sentences))} unique sentences in batches of {batch_size} ...")

        sentence_embeddings = _embed_sentences(
            sentences=all_sentences,
            model=model,
            api_key=api_key,
            task_type=task_type,
            batch_size=batch_size,
            requests_per_minute=requests_per_minute,
        )

        # Compute cosine similarity for each pair
        cos_sims = []
        for _, row in gold_df.iterrows():
            vec_a = np.array(sentence_embeddings[row[col_a]]).reshape(1, -1)
            vec_b = np.array(sentence_embeddings[row[col_b]]).reshape(1, -1)
            cos_sims.append(float(cosine_similarity(vec_a, vec_b)[0, 0]))

        scores_df = pd.DataFrame({
            "ID": gold_df["ID"].values,
            "embedding_score": cos_sims,
        })
        scores_df.to_csv(scores_csv, index=False)
        print(f"Scores saved to: {scores_csv}")

    # ── correlate with gold ───────────────────────────────────────────────────
    merged = pd.merge(
        gold_df[["ID", "score"]].rename(columns={"score": "gold_score"}),
        scores_df[["ID", "embedding_score"]],
        on="ID",
        how="inner",
    ).dropna(subset=["gold_score", "embedding_score"])

    if len(merged) < 2:
        raise RuntimeError(
            f"Too few matching pairs for correlation ({len(merged)}). "
            "Check that IDs in the scores CSV match the gold data."
        )

    pearson_r, pearson_p = pearsonr(merged["gold_score"], merged["embedding_score"])
    spearman_r, spearman_p = spearmanr(merged["gold_score"], merged["embedding_score"])
    kendall_r, kendall_p = kendalltau(merged["gold_score"], merged["embedding_score"])

    print(f"\n{'=' * 60}")
    print(f"Model    : {model}")
    print(f"Config   : {config_path}")
    print(f"Task     : {task_type}")
    print(f"Pairs    : {len(merged)} / {len(gold_df)} scored")
    print(f"Pearson  : {pearson_r:.4f}  (p={pearson_p:.4g})")
    print(f"Spearman : {spearman_r:.4f}  (p={spearman_p:.4g})")
    print(f"Kendall  : {kendall_r:.4f}  (p={kendall_p:.4g})")
    print(f"Output   : {output_dir}")
    print(f"{'=' * 60}")

    results_row = {
        "config_path": config_path,
        "model": model,
        "task_type": task_type,
        "gold_data": Path(gold_data_path).name,
        "text_cols": "unicode" if use_unicode else "EWTS",
        "n_pairs_scored": len(merged),
        "n_pairs_total": len(gold_df),
        "pearson": round(float(pearson_r), 6),
        "pearson_p": round(float(pearson_p), 6),
        "spearman": round(float(spearman_r), 6),
        "spearman_p": round(float(spearman_p), 6),
        "kendall": round(float(kendall_r), 6),
        "kendall_p": round(float(kendall_p), 6),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
    }

    results_csv = output_dir / "eval_gold_results.csv"
    new_row_df = pd.DataFrame([results_row])
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        new_row_df = pd.concat([existing, new_row_df], ignore_index=True)
    new_row_df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")

    return pd.DataFrame([results_row])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Gemini embeddings on gold similarity data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", dest="config_path", required=True,
        help="Embedding model config YAML (e.g. supported_models_config/gemini_embedding_001_config.yaml).",
    )
    p.add_argument(
        "--gold-data", dest="gold_data_path", default=None,
        help=f"Path to the gold .xlsx (default: data/{DEFAULT_GOLD_FILENAME}).",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory for result files (default: results/eval_gold_embedding/<timestamp>/).",
    )
    p.add_argument(
        "--use-unicode", dest="use_unicode", action="store_true",
        help="Use SentenceA/B_unicode columns instead of EWTS.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip embedding if embedding_scores.csv already exists in --output-dir.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    eval_gold_embedding(
        config_path=args.config_path,
        gold_data_path=args.gold_data_path,
        output_dir=args.output_dir,
        use_unicode=args.use_unicode,
        resume=args.resume,
    )
