"""
Train mBERT with MultipleNegativesRankingLoss on consecutive segment pairs,
then evaluate on the four sets (A, B, C, D).

Functions
---------
generate_consecutive_pairs  -- build (anchor, positive) CSV from raw segments
train_mnrl_and_evaluate     -- run the full MNRL training + evaluation pipeline

Usage:
    python -m sub_tasks.mnrl_training
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sub_tasks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sensim/

from app_config import AppConfig
from common_utils import load_dataframe, load_yaml
import aggregate_sets
import sbert_with_pretrain


def generate_consecutive_pairs(segments_filepath, output_filepath):
    """
    Generate (anchor, positive) pairs from consecutive text segments.
    Consecutive segments within the same Source_Line_Number are paired.
    Uses the Segmented_Text_EWTS column.
    """
    df = load_dataframe(segments_filepath)
    df = df.sort_values(['Source_Line_Number', 'Sentence_Order']).reset_index(drop=True)

    pairs = []
    for _, group in df.groupby('Source_Line_Number'):
        texts = group['Segmented_Text_EWTS'].tolist()
        for i in range(len(texts) - 1):
            if pd.notna(texts[i]) and pd.notna(texts[i + 1]):
                pairs.append({'SentenceA': texts[i], 'SentenceB': texts[i + 1]})

    pairs_df = pd.DataFrame(pairs)
    filepath = Path(segments_filepath)
    if filepath.suffix.lower() == '.xlsx':
        pairs_df.to_excel(output_filepath, index=False)
    elif filepath.suffix.lower() == '.csv':
        pairs_df.to_csv(output_filepath, index=False)
    else:
        raise ValueError(f"segments_filepath: {segments_filepath} with unsupported extention")
    print(f"Generated {len(pairs_df)} consecutive segment pairs -> {output_filepath}")
    return output_filepath


def train_mnrl_and_evaluate(run_time_identifier=None):
    """
    Train Intellexus/mbert-tibetan-continual-wylie-final with MultipleNegativesRankingLoss
    on consecutive segment pairs, then evaluate on the four sets (A, B, C, D).
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
    train_pairs_filename = "consecutive_segment_pairs_v4.csv"
    results_filename = f"mnrl_sets_results_{run_time_identifier}.csv"
    aggregation_filepath = os.path.join(sensim_base, "results", f"mnrl_aggregated_sets_results_{run_time_identifier}.csv")
    model_dir = os.path.join(sensim_base, f"ckpts/sts-b/cino/{run_time_identifier}")

    batch_size = '128'
    train_args = [
        #"--hf_base_model", "Intellexus/mbert-tibetan-continual-wylie-final",
        "--hf_base_model", "hfl/cino-large-v2",
        "--hf_token", hf_token,
        "--pooling_strategy", "weightedmean",
        "--epochs", "3",
        "--learning_rate", "2e-5",
        "--data_dir", data_dir,
        "--train_dir", data_dir,
        '--results_dir', results_dir,
        "--model_dir", model_dir,
        "--train_filenames", train_pairs_filename,
        "--results_filename", results_filename,
        "--loss_type", "mnrl",
        "--batch_size", batch_size,
        "--keep_previous_model_in_dir",
        "--no_eval",
        "--gradient_accumulation_steps", "4",
    ]

    print(f"\n{'=' * 60}")
    print(f"Training MNRL (no validation / test evaluation)")
    print(f"{'=' * 60}")
    sbert_with_pretrain.main(train_args)


if __name__ == "__main__":
    run_time_identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_mnrl_and_evaluate(run_time_identifier=run_time_identifier)
    # generate_consecutive_pairs(
    #     '/home/shailu1492/repositories/intellexus-model/sensim/data/merged_kangyur_tengyur_segments_v4.csv',
    #     '/home/shailu1492/repositories/intellexus-model/sensim/data/consecutive_segment_pairs_v4',
    # )
