"""
Train mBERT with ContrastiveTensionLossInBatchNegatives.

Each segment is used as both SentenceA and SentenceB. The two forward passes use
different dropout masks, so the model learns to produce consistent embeddings for
the same text while treating other in-batch sentences as negatives.

Usage:
    python -m sub_tasks.ct_training
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sub_tasks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sensim/

from app_config import AppConfig
from common_utils import load_yaml, load_dataframe
import sbert_with_pretrain

_SEGMENTS_FILENAME = "merged_kangyur_tengyur_segments_v4.csv"
_PAIRS_FILENAME = "ct_self_pairs_v4.csv"


def generate_self_pairs(segments_filepath, output_filepath):
    """
    Build a pairs file where SentenceA == SentenceB for every segment.
    ContrastiveTensionLoss relies on dropout to produce two different representations
    of the same sentence, so no label is needed.
    """
    df = load_dataframe(segments_filepath)
    sentences = df["Segmented_Text_EWTS"].dropna().unique()
    pairs_df = pd.DataFrame({"SentenceA": sentences, "SentenceB": sentences})
    pairs_df.to_csv(output_filepath, index=False)
    print(f"Generated {len(pairs_df)} self-pairs -> {output_filepath}")
    return output_filepath


def train_ct_and_evaluate(run_time_identifier=None):
    """
    Train with ContrastiveTensionLossInBatchNegatives using self-paired segments.
    """
    app_config = AppConfig()
    _fallback_base = str(Path(__file__).resolve().parent.parent)
    sensim_base = str(Path(app_config.get("sensim_base_dir", _fallback_base)).resolve())

    keys = load_yaml(os.path.join(sensim_base, "keys.yaml"), must_exist=True)
    if "HF_TOKEN" in keys:
        hf_token = str(keys["HF_TOKEN"])
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    data_dir = os.path.join(sensim_base, "data")
    results_dir = os.path.join(sensim_base, "results")

    segments_filepath = os.path.join(data_dir, _SEGMENTS_FILENAME)
    train_pairs_filepath = os.path.join(data_dir, _PAIRS_FILENAME)

    if not os.path.exists(train_pairs_filepath):
        print(f"Generating self-pairs from {segments_filepath} ...")
        generate_self_pairs(segments_filepath, train_pairs_filepath)

    results_filename = f"ct_sets_results_{run_time_identifier}.csv"
    model_dir = os.path.join(sensim_base, f"ckpts/sts-b/modernbert/ct_{run_time_identifier}")

    batch_size = '128'
    train_args = [
        "--hf_base_model", "Intellexus/IntellexusBert-2.0",
        "--hf_token", hf_token,
        "--pooling_strategy", "weightedmean",
        "--epochs", "3",
        "--learning_rate", "2e-5",
        "--data_dir", data_dir,
        "--train_dir", data_dir,
        "--results_dir", results_dir,
        "--model_dir", model_dir,
        "--train_filenames", _PAIRS_FILENAME,
        "--results_filename", results_filename,
        "--loss_type", "ct",
        "--batch_size", batch_size,
        "--keep_previous_model_in_dir",
        "--no_eval",
        "--gradient_accumulation_steps", "4",
    ]

    print(f"\n{'=' * 60}")
    print(f"Training ContrastiveTensionLossInBatchNegatives (self-pairs), model will be saved to {model_dir}, results to {os.path.join(results_dir, results_filename)}")
    print(f"{'=' * 60}")
    sbert_with_pretrain.main(train_args)

    print(f"\n{'=' * 60}")
    print(f"\nTraining complete. Results saved to {os.path.join(results_dir, results_filename)}, model saved to {model_dir}")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    run_time_identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_ct_and_evaluate(run_time_identifier=run_time_identifier)
