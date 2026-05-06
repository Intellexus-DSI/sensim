"""
Analyse wrong_pairs_log.csv and surface the most disagreed pairs.

Score per unique pair:
    score = count × mean(|error|)

Where:
  - count      = number of times the pair was logged (across runs / datasets)
  - mean |error| = average absolute difference between predicted_cosine and label

Usage:
    python -m sub_tasks.wrong_pairs_top                          # defaults
    python -m sub_tasks.wrong_pairs_top --top 20 --input /path/to/wrong_pairs_log.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sensim/


def top_wrong_pairs(
    input_file: str,
    output_file: str,
    top: int = 10,
) -> pd.DataFrame:
    """
    Read *input_file*, compute a disagreement score for every unique (text1, text2)
    pair, and write the top-*top* rows to *output_file*.

    Score = count × mean(|error|)

    Returns the top-N DataFrame.
    """
    df = pd.read_csv(input_file)

    required = {"text1", "text2", "error", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    df["abs_error"] = df["error"].abs()

    agg = (
        df.groupby(["text1", "text2"], sort=False)
        .agg(
            count=("abs_error", "count"),
            mean_abs_error=("abs_error", "mean"),
            mean_label=("label", "mean"),
            mean_predicted_cosine=("predicted_cosine", "mean"),
            datasets=("dataset", lambda s: ", ".join(sorted(s.unique()))),
            run_timestamps=("run_timestamp", lambda s: ", ".join(sorted(s.unique()))),
        )
        .reset_index()
    )

    agg["score"] = agg["count"] * agg["mean_abs_error"]
    agg = agg.sort_values("score", ascending=False).head(top).reset_index(drop=True)

    agg["mean_abs_error"] = agg["mean_abs_error"].round(4)
    agg["mean_label"] = agg["mean_label"].round(4)
    agg["mean_predicted_cosine"] = agg["mean_predicted_cosine"].round(4)
    agg["score"] = agg["score"].round(4)

    col_order = [
        "score", "count", "mean_abs_error",
        "mean_label", "mean_predicted_cosine",
        "text1", "text2",
        "datasets", "run_timestamps",
    ]
    agg = agg[col_order]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_file, index=False)
    print(f"Top-{top} most disagreed pairs saved to: {output_file}")

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface the most disagreed pairs from wrong_pairs_log.csv")
    parser.add_argument(
        "--input",
        default="/home/shailu1492/repositories/intellexus-model/sensim/results/wrong_pairs_log.csv",
        help="Path to wrong_pairs_log.csv",
    )
    parser.add_argument(
        "--output",
        default="/home/shailu1492/repositories/intellexus-model/sensim/results/wrong_pairs_top.csv",
        help="Path to write the top-N output CSV",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top pairs to keep (default: 10)",
    )
    args = parser.parse_args()

    result = top_wrong_pairs(
        input_file=args.input,
        output_file=args.output,
        top=args.top,
    )
    print(result.to_string(index=False))
