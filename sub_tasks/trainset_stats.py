"""
Analyse train_backups files and report FaissDistance and score statistics.

For each .xlsx file in the train_backups folder, computes mean and std of
the FaissDistance and score columns and displays a summary DataFrame.

Usage:
    python -m sub_tasks.trainset_stats
    python -m sub_tasks.trainset_stats --folder /path/to/train_backups
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sensim/

TRAIN_BACKUPS_DIR = Path(__file__).resolve().parent.parent / "train_backups"


def analyze_trainset_files(folder: str = None) -> pd.DataFrame:
    """
    Iterate over all .xlsx files in *folder*, compute mean and std of
    FaissDistance and score, and return a summary DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        file, faiss_distance_mean, faiss_distance_std, score_mean, score_std
    """
    folder_path = Path(folder) if folder else TRAIN_BACKUPS_DIR

    xlsx_files = sorted(folder_path.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in: {folder_path}")

    rows = []
    for path in xlsx_files:
        df = pd.read_excel(path)

        missing = {"FaissDistance", "score"} - set(df.columns)
        if missing:
            print(f"[WARN] {path.name}: missing columns {missing}, skipping.")
            continue

        rows.append(
            {
                "file": path.name,
                "faiss_distance_mean": round(df["FaissDistance"].mean(), 4),
                "faiss_distance_std": round(df["FaissDistance"].std(), 4),
                "score_mean": round(df["score"].mean(), 4),
                "score_std": round(df["score"].std(), 4),
            }
        )

    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show FaissDistance and score stats for each train_backups file"
    )
    parser.add_argument(
        "--folder",
        default=None,
        help=f"Path to the train_backups folder (default: {TRAIN_BACKUPS_DIR})",
    )
    args = parser.parse_args()

    result = analyze_trainset_files(folder=args.folder)
    print(result.to_string(index=False))
