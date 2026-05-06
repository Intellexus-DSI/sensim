from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import os

from common_utils import setup_logging, install_global_exception_logging
from models import SBERTModel
from training_files_manager import TrainingFilesManager
from app_config import AppConfig

LOGGER = logging.getLogger(__name__)

app_config = AppConfig()


def _build_arg_parser() -> ArgumentParser:
    p = ArgumentParser(description="Run the unsupervised training pipline.")

    p.add_argument("--segments-dataset", type=str, default="merged_tibetan_segments.xlsx",
                   help="Path to segments dataset (default: merged_tibetan_segments.xlsx)", dest="segments_dataset")
    p.add_argument("--repo-id", type=str, required=True,
                   help="HuggingFace repository ID to publish the model to (e.g., username/repo_name)", dest="repo_id")
    p.add_argument("--commit-message", type=str, default="Upload unsupervised trained model",
                   help="Commit message for the model upload ", dest="commit_message")

    # -------------------------
    # Logging configuration
    # -------------------------
    p.add_argument("--log-dir", type=str, default="./logs", help="Directory for log files (default: ./logs)")
    p.add_argument("--log-file", type=str, default="", help="Log filename. If empty, auto-generate one.")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    # Additional arguments
    p.add_argument("--no-console-log", action="store_true", help="Disable console logging (file only)")
    return p


def _parse_args(argv: Optional[List[str]]) -> Namespace:
    p = _build_arg_parser()
    args, _ = p.parse_known_args(argv)
    return args


def unsupervised_training_pipeline(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the active sampling pipeline.

    IMPORTANT: This function refactors structure, logging, and progress reporting,
    while preserving the original pipeline logic and file outputs.
    """
    args = _parse_args(argv)

    base_dir = app_config.get('sensim_base_dir', '.')
    log_dir = app_config.get('log_dir', os.path.join(base_dir, 'logs'))

    log_path = setup_logging(
        log_dir=log_dir,
        log_file=args.log_file,
        log_level=args.log_level,
        console_enabled=(not args.no_console_log),
        name=__name__,
    )
    install_global_exception_logging(LOGGER)

    LOGGER.info("Starting unsupervised training. argv=%s", argv)
    LOGGER.info("Log file: %s", log_path)

    validation_ds = "train_A_300_scored.xlsx"
    test_ds = ["train_B_300_scored.xlsx", "test_pairs_300_scored.xlsx", "validation_pairs_100_scored.xlsx"]
    segments_dataset = args.segments_dataset
    repo_id = args.repo_id
    if not repo_id:
        LOGGER.error("HuggingFace repo ID is required. Use --repo-id to specify it.")
        raise ValueError("HuggingFace repo ID is required.")
    commit_message = args.commit_message

    files_manager = TrainingFilesManager(
        models=[],
        validation_dataset=validation_ds,
        test_dataset=test_ds,
        segments_dataset=segments_dataset,
    )

    # -------------------------
    # Unsupervised Training and Publishing
    # -------------------------
    model = SBERTModel(config_path="config.yaml", argv=argv, files_manager=files_manager)
    model.unsupervised_train()
    model.publish_model(repo_id=repo_id, commit_message=commit_message)

    LOGGER.info("Unsupervised training completed.")


def main() -> int:
    try:
        unsupervised_training_pipeline(sys.argv[1:])
        return 0
    except Exception:
        LOGGER.exception("Fatal error. Exiting.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
