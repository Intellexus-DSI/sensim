"""
Upload a local SentenceTransformer checkpoint to Hugging Face Hub.

Usage:
    python -m sub_tasks.model_tasks \
        --folder results/active_sampling/2026-02-06_17-38-03/ckpts/sts-b \
        --repo-id YourOrg/your-model-name

    # With a custom commit message:
    python -m sub_tasks.model_tasks \
        --folder results/active_sampling/2026-02-06_17-38-03/ckpts/sts-b \
        --repo-id YourOrg/your-model-name \
        --commit-message "Upload active-sampling iteration 10 checkpoint"
"""

import argparse
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aggregate_sets
import generate_trainset
import sbert_with_pretrain

from huggingface_hub import HfApi, ModelCard, ModelCardData, login

from common_utils import load_yaml
from app_config import AppConfig


def _resolve_keys_path(keys_path: str) -> Path:
    """Return the first existing candidate for keys_path, searching common locations."""
    candidates = [
        Path(keys_path),
        Path(__file__).resolve().parent.parent / keys_path,
        Path(__file__).resolve().parent.parent / "keys.yaml",
    ]
    for p in candidates:
        if p.expanduser().exists():
            return p.expanduser()
    raise FileNotFoundError(
        f"keys.yaml not found. Tried:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def upload_model(folder: str, repo_id: str, commit_message: str, keys_path: str,
                 license: str | None = None):
    keys = load_yaml(str(_resolve_keys_path(keys_path)))
    hf_token = keys.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not found in keys file.")

    login(token=hf_token)

    folder_path = Path(folder)
    if not folder_path.exists():
        sys.exit(f"Folder does not exist: {folder_path}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns="./checkpoint-*", # Optionally delete old checkpoints in the repo to save space. Adjust the pattern as needed. Need to be tested!
    )
    print(f"Uploaded {folder_path} -> https://huggingface.co/{repo_id}")

    if license:
        card_data = ModelCardData(license=license)
        card = ModelCard.from_template(card_data)
        card.push_to_hub(repo_id, commit_message=f"Set license: {license}")
        print(f"License set to: {license}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload a model checkpoint to Hugging Face Hub.")
    parser.add_argument("--folder", type=str,
                        default="",
                        help="Path to the model folder to upload")
    parser.add_argument("--repo-id", type=str, dest="repo_id",
                        help="Hugging Face repo id (e.g. YourOrg/model-name)")
    parser.add_argument("--commit-message", type=str, default="Upload model checkpoint",
                        dest="commit_message", help="Commit message for the upload")
    parser.add_argument("--keys", type=str, default="../keys.yaml",
                        help="Path to keys YAML containing HF_TOKEN (default: keys.yaml)")
    parser.add_argument("--license", type=str, default=None,
                        help="SPDX license identifier to set on the model card (e.g. apache-2.0)")
    args = parser.parse_args(argv)

    upload_model(args.folder, args.repo_id, args.commit_message, args.keys,
                 license=args.license)

def train_and_score_model_with_mined_pairs(run_time_identifier=None):
    config = AppConfig()
    base_dir = config.get("sensim_base_dir")
    RESULTS_FOLDER = f"{base_dir}/results"
    # --- CONFIGURATION ---
    BASE_FOLDER = f"{RESULTS_FOLDER}/active_sampling/{run_time_identifier}"
    output_filename = f"merged_llms_scored_{run_time_identifier}.xlsx"
    sets_filename = f"llms_sets_results_{run_time_identifier}.csv"
    sets_filepath = f"{BASE_FOLDER}/{sets_filename}"
    aggregation_filepath = f"{RESULTS_FOLDER}/active_sampling/{run_time_identifier}/llms_aggregated_sets_results_{run_time_identifier}.csv"

    generate_trainset.merge_excel_files(BASE_FOLDER, output_filename)

    do_copy_to_data_folder = True  # Set to True to copy the merged file to the data folder
    if do_copy_to_data_folder:
        generate_trainset.copy_to_data_folder(BASE_FOLDER, output_filename)

    # Define the validation and test filenames for each set
    validation_filenames = [
        "validation_pairs_A_shuffled_150_scored.xlsx",
        "validation_pairs_B_shuffled_150_scored.xlsx",
        "validation_pairs_C_shuffled_150_scored.xlsx",
        "validation_pairs_D_shuffled_150_scored.xlsx"
    ]
    # These test files should already exist in the data folder, so we don't need to copy them
    test_filenames = [
        "test_pairs_A_shuffled_250_scored.xlsx",
        "test_pairs_B_shuffled_250_scored.xlsx",
        "test_pairs_C_shuffled_250_scored.xlsx",
        "test_pairs_D_shuffled_250_scored.xlsx"
    ]

    # Define the common training arguments for all sets
    train_args = [
        "--hf_base_model", "Intellexus/mbert-tibetan-continual-wylie-final",
        "--pooling_strategy", "weightedmean",
        "--epochs", "7",
        "--learning_rate", "2e-5",
        "--data_dir", "../data/NewDataA-D",
        "--train_dir", "../data/NewDataA-D",
        "--train_filenames", output_filename,
        "--results_filename", sets_filename,
        "--model_dir", "ckpts/sts-b/synthetic",
        "--loss_type", "cosent",
        "--results_dir", BASE_FOLDER,
    ]

    # Loop through each validation and test file pair, train the model, and score it
    for val_file, test_file in zip(validation_filenames, test_filenames):

        loop_args = train_args + [
            "--validation_filename", val_file,
            "--test_filenames", test_file,
        ]
        print("loop_args", loop_args)
        sbert_with_pretrain.main(loop_args)
        print(f"Completed training and scoring for validation: {val_file} and test: {test_file} and save results to {sets_filename}")

    # After training and scoring, copy the sets results to the base folder and aggregate them
    #shutil.copy(sets_filepath, BASE_FOLDER)
    print(f"aggregation_filepath: {aggregation_filepath}")
    aggregate_sets.aggregate_sets(sets_filepath, aggregation_filepath)



if __name__ == "__main__":
    main()
    
    # train_and_score_model_with_mined_pairs(run_time_identifier=run_time_identifier)
    #folder = '/home/shailu1492/repositories/intellexus-model/sensim/ckpts/sts-b/modernbert/Tib-Bi-Dharma'
    #repo_id = 'Intellexus/Tib-Bi-Dharma-EWTS'
    #keys_path='/home/shailu1492/repositories/intellexus-model/sensim/keys.yaml'
    #commit_message = f"This model is a Classical Tibetan sentence embedder built upon the ModernBERT architecture. It is initialized from the unsupervised tibetan-modernbert-v4-b64-consecutive-segments checkpoint (derived from Intellexus/IntellexusBert-2.0). The model was fine-tuned using a sequential two-stage pipeline: it was first aligned on a synthetically mined Best-Worst Scaling (BWS) annotated set (version 2026-02-27_22-20-37), and subsequently refined using human-annotated Gold data. Evaluated via rigorous 4-fold cross-validation, the model achieves a peak Spearman correlation of 0.864 for cosine similarity."
    #upload_model(folder=folder, repo_id=repo_id, commit_message=commit_message, keys_path=keys_path)


