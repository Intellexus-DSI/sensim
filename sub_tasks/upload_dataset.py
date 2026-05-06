"""
Upload a dataset file to Hugging Face Hub as a HF Dataset.

Usage:
    python -m sub_tasks.upload_dataset \
        --repo-id YourOrg/your-dataset-name

    # Custom file, private, drop columns, sort before uploading:
    python -m sub_tasks.upload_dataset \
        --file data/NewDataA-D/merged_trainset_2026-02-27_12-24-40.xlsx \
        --repo-id YourOrg/your-dataset-name \
        --private \
        --drop-columns FaissDistance is_unicode_score \
        --order-by score \
        --rename-columns SentenceA=source SentenceB=target
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common_utils import load_yaml

DEFAULT_FILE = "data/all_gold_pairs_1000_scored.xlsx"


def preprocess_df(df: pd.DataFrame, drop_columns: list[str], order_by: list[str],
                  rename_columns: dict[str, str]) -> pd.DataFrame:
    if drop_columns:
        missing = [c for c in drop_columns if c not in df.columns]
        if missing:
            sys.exit(f"Columns to drop not found in dataset: {missing}")
        df = df.drop(columns=drop_columns)
        print(f"Dropped columns: {drop_columns}")

    if order_by:
        missing = [c for c in order_by if c not in df.columns]
        if missing:
            sys.exit(f"Order-by columns not found in dataset: {missing}")
        df = df.sort_values(by=order_by).reset_index(drop=True)
        print(f"Sorted by: {order_by}")

    if rename_columns:
        missing = [c for c in rename_columns if c not in df.columns]
        if missing:
            sys.exit(f"Columns to rename not found in dataset: {missing}")
        df = df.rename(columns=rename_columns)
        print(f"Renamed columns: {rename_columns}")

    return df


def upload_dataset(file: str, repo_id: str, private: bool, commit_message: str, keys_path: str,
                   drop_columns: list[str] = None, order_by: list[str] = None,
                   rename_columns: dict[str, str] = None):
    keys = load_yaml(keys_path)
    hf_token = keys.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not found in keys file.")

    login(token=hf_token)

    file_path = Path(file)
    if not file_path.exists():
        sys.exit(f"File does not exist: {file_path}")

    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path)
    df = preprocess_df(df, drop_columns or [], order_by or [], rename_columns or {})
    dataset = Dataset.from_pandas(df, preserve_index=False)

    print(f"Pushing {len(dataset)} rows to {repo_id}...")
    dataset.push_to_hub(repo_id, private=private, commit_message=commit_message)
    print(f"Done -> https://huggingface.co/datasets/{repo_id}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload an xlsx dataset to Hugging Face Hub.")
    parser.add_argument("--file", type=str, default=DEFAULT_FILE,
                        help=f"Path to the xlsx file to upload (default: {DEFAULT_FILE})")
    parser.add_argument("--repo-id", type=str, dest="repo_id", required=True,
                        help="Hugging Face dataset repo id (e.g. YourOrg/dataset-name)")
    parser.add_argument("--private", action="store_true", default=False,
                        help="Create the repo as private (default: public)")
    parser.add_argument("--commit-message", type=str, default="Upload dataset",
                        dest="commit_message", help="Commit message for the upload")
    parser.add_argument("--drop-columns", type=str, nargs="+", default=None, dest="drop_columns",
                        metavar="COL", help="Columns to remove before uploading (space-separated)")
    parser.add_argument("--order-by", type=str, nargs="+", default=None, dest="order_by",
                        metavar="COL", help="Columns to sort by before uploading (space-separated)")
    parser.add_argument("--rename-columns", type=str, nargs="+", default=None, dest="rename_columns",
                        metavar="OLD=NEW", help="Columns to rename before uploading (e.g. old_name=new_name)")
    parser.add_argument("--keys", type=str, default="keys.yaml",
                        help="Path to keys YAML containing HF_TOKEN (default: keys.yaml)")
    args = parser.parse_args(argv)

    rename_columns = {}
    if args.rename_columns:
        for pair in args.rename_columns:
            if "=" not in pair:
                sys.exit(f"Invalid --rename-columns format '{pair}', expected OLD=NEW")
            old, new = pair.split("=", 1)
            rename_columns[old] = new

    upload_dataset(args.file, args.repo_id, args.private, args.commit_message, args.keys,
                   drop_columns=args.drop_columns, order_by=args.order_by, rename_columns=rename_columns)


if __name__ == "__main__":
    main()
