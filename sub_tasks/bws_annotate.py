"""
BWS (Best-Worst Scaling) annotation and scoring using embedding cosine similarity.

Commands:
  annotate  - Use an embedding model to mark best/worst pairs in 4-tuple rows.
  score     - Compute BWS scores from an annotated file (no model needed).

Usage:
    python -m sub_tasks.bws_annotate annotate \
        --input_file data/data_4_pairs/llms_4_pair_annotations_valid_A_4_pairs_100.xlsx \
        --output_file data/data_4_pairs/llms_4_pair_annotations_valid_A_4_pairs_100_annotated.xlsx

    python -m sub_tasks.bws_annotate score \
        --input_file data/data_4_pairs/llms_4_pair_annotations_valid_A_4_pairs_100_annotated.xlsx \
        --scores_file data/data_4_pairs/bws_scores.xlsx
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PAIR_INDICES = [1, 2, 3, 4]


def collect_unique_sentences(df):
    """Collect all unique sentences from pair columns."""
    sentences = set()
    for idx in PAIR_INDICES:
        for suffix in ("A", "B"):
            col = f"pair_{idx}_{suffix}"
            sentences.update(df[col].dropna().astype(str).tolist())
    return sentences


def encode_sentences(model, sentences, batch_size):
    """Encode all unique sentences and return a dict mapping sentence -> embedding."""
    sentence_list = sorted(sentences)
    print(f"Encoding {len(sentence_list)} unique sentences...")
    embeddings = model.encode(sentence_list, batch_size=batch_size, show_progress_bar=True)
    return dict(zip(sentence_list, embeddings))


def annotate_bws(df, embedding_cache):
    """Annotate each row with best_pair and worst_pair based on cosine similarity."""
    all_cosines = []

    for row_idx, row in df.iterrows():
        pair_cosines = {}
        for idx in PAIR_INDICES:
            sent_a = str(row[f"pair_{idx}_A"])
            sent_b = str(row[f"pair_{idx}_B"])
            pair_id = row[f"id_{idx}"]

            emb_a = embedding_cache[sent_a].reshape(1, -1)
            emb_b = embedding_cache[sent_b].reshape(1, -1)
            cosine_sim = 1 - paired_cosine_distances(emb_a, emb_b)[0]
            pair_cosines[pair_id] = cosine_sim

        best_id = max(pair_cosines, key=pair_cosines.get)
        worst_id = min(pair_cosines, key=pair_cosines.get)

        df.at[row_idx, "best_pair"] = best_id
        df.at[row_idx, "worst_pair"] = worst_id

        all_cosines.append(list(pair_cosines.values()))

    all_cosines = np.array(all_cosines)
    print(f"\nCosine similarity stats across all pairs:")
    print(f"  Mean: {all_cosines.mean():.4f}")
    print(f"  Std:  {all_cosines.std():.4f}")
    print(f"  Min:  {all_cosines.min():.4f}")
    print(f"  Max:  {all_cosines.max():.4f}")

    return df


def compute_bws_scores(df):
    """Compute BWS scores from annotated 4-tuple data.

    For each pair ID, counts how many times it was chosen as best vs worst
    across all tuples where it appears:
        raw_score = (best_count - worst_count) / total_appearances   (range [-1, 1])
        score     = (raw_score + 1) / 2                              (range [0, 1])
    """
    best_counts = Counter(df["best_pair"].dropna().astype(str))
    worst_counts = Counter(df["worst_pair"].dropna().astype(str))

    # Count total appearances per ID across all tuple positions
    appearances = Counter()
    for idx in PAIR_INDICES:
        appearances.update(df[f"id_{idx}"].dropna().astype(str))

    # Build lookup: pair ID -> (SentenceA, SentenceB)
    pair_sentences = {}
    for _, row in df.iterrows():
        for idx in PAIR_INDICES:
            pid = str(row[f"id_{idx}"]) if pd.notna(row[f"id_{idx}"]) else None
            if pid and pid not in pair_sentences:
                pair_sentences[pid] = (str(row[f"pair_{idx}_A"]), str(row[f"pair_{idx}_B"]))

    rows = []
    for pair_id, total in sorted(appearances.items()):
        best = best_counts.get(pair_id, 0)
        worst = worst_counts.get(pair_id, 0)
        raw_score = (best - worst) / total
        score = (raw_score + 1) / 2
        sent_a, sent_b = pair_sentences.get(pair_id, ("", ""))
        rows.append({
            "ID": pair_id,
            "SentenceA": sent_a,
            "SentenceB": sent_b,
            "best_count": best,
            "worst_count": worst,
            "appearances": total,
            "raw_score": round(raw_score, 4),
            "score": round(score, 4),
        })

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\nBWS Scores ({len(scores_df)} pairs):")
    print(f"  Score range: [{scores_df['score'].min():.4f}, {scores_df['score'].max():.4f}]")
    print(f"  Mean score:  {scores_df['score'].mean():.4f}")
    print(f"  Median:      {scores_df['score'].median():.4f}")

    return scores_df


def cmd_annotate(args):
    """Annotate subcommand: load model, compute similarities, mark best/worst."""
    from sentence_transformers import SentenceTransformer
    import model_utils

    output_file = args.output_file or args.input_file

    df = pd.read_excel(args.input_file, engine="openpyxl")
    df["best_pair"] = df["best_pair"].astype(object)
    df["worst_pair"] = df["worst_pair"].astype(object)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    gpu_device = model_utils.get_gpu_device()
    print(f"Loading model: {args.model_name} on {gpu_device}")
    model = SentenceTransformer(args.model_name, device=gpu_device)
    print("Model loaded.")

    unique_sentences = collect_unique_sentences(df)
    embedding_cache = encode_sentences(model, unique_sentences, args.batch_size)

    df = annotate_bws(df, embedding_cache)

    df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"\nAnnotated file saved to: {output_file}")
    print(f"Annotated {df['best_pair'].notna().sum()}/{len(df)} rows.")


def cmd_score(args):
    """Score subcommand: compute BWS scores from an already-annotated file."""
    df = pd.read_excel(args.input_file, engine="openpyxl")
    print(f"Loaded {len(df)} annotated rows from {args.input_file}")

    missing = df["best_pair"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have no best_pair annotation — skipping those.")

    scores_df = compute_bws_scores(df)

    scores_df.to_excel(args.scores_file, index=False, engine="openpyxl")
    print(f"\nScores saved to: {args.scores_file}")

    print(f"\nTop 10 pairs:")
    print(scores_df.head(10).to_string(index=False))
    print(f"\nBottom 10 pairs:")
    print(scores_df.tail(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="BWS annotation and scoring")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # annotate subcommand
    p_ann = subparsers.add_parser("annotate", help="Annotate 4-tuples with best/worst using an embedding model")
    p_ann.add_argument("--input_file", required=True, help="Path to the 4-tuple Excel file")
    p_ann.add_argument("--output_file", default=None, help="Output path (defaults to overwriting input)")
    p_ann.add_argument("--model_name", default="Qwen/Qwen3-Embedding-8B", help="SentenceTransformer model name")
    p_ann.add_argument("--batch_size", type=int, default=32, help="Encoding batch size")

    # score subcommand
    p_score = subparsers.add_parser("score", help="Compute BWS scores from annotated file")
    p_score.add_argument("--input_file", required=True, help="Path to the annotated 4-tuple Excel file")
    p_score.add_argument("--scores_file", required=True, help="Output path for the scores Excel file")

    args = parser.parse_args()

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "score":
        cmd_score(args)


if __name__ == "__main__":
    main()
