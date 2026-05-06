from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from training_files_manager import TrainingFilesManager
from .load_dataframe import load_dataframe
from .save_dataframe import save_dataframe_single

LOGGER = logging.getLogger(__name__)


def concatenate_files(files_manager: TrainingFilesManager) -> None:
    """Concatenate per-model similarity outputs into a single file.

    Logic preserved: load each model's current results, concat row-wise, save to
    `files_manager.llms_4_pair_annotations_current`.
    """
    dfs = []
    for model_name in files_manager.models:
        path = files_manager.model_similarity_results_current(model_name)
        LOGGER.info("Loading model results: %s -> %s", model_name, path)
        dfs.append(load_dataframe(path))

    if not dfs:
        raise ValueError("No model results found to concatenate (files_manager.models is empty).")

    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = files_manager.llms_4_pair_annotations_current
    LOGGER.info("Saving concatenated results (%d rows) -> %s", len(combined_df), out_path)
    save_dataframe_single(combined_df, out_path, exists_ok=True)
