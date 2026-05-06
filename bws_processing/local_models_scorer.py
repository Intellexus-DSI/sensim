from dataclasses import dataclass
from typing import Optional, List, Any
import logging

import gc
import numpy as np
import torch
import pandas as pd

from common_utils import load_dataframe, save_dataframe_single
from training_files_manager import TrainingFilesManager
from app_config import AppConfig

LOGGER = logging.getLogger(__name__)


def _normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


@dataclass
class LocalModelsScorer:
    _files_manager: TrainingFilesManager
    _ensemble: list

    app_config = AppConfig()

    def __init__(
        self,
        files_manager: TrainingFilesManager,
        config_path: Optional[str] = None,
        argv: Optional[List[str]] = None,
        **overrides: Any,
    ) -> None:
        self._files_manager = files_manager
        self._ensemble = list(
            self.app_config.get("local_models_ensemble", [])
        )
        if not self._ensemble:
            raise ValueError(
                "local_models_ensemble is empty or missing in config.yaml"
            )
        self._use_unicode = bool(self.app_config.get("local_models_scorer_use_unicode", False))
        self._ewts_col = str(self.app_config.get("sentence_col", "Segmented_Text_EWTS"))
        self._unicode_col = str(self.app_config.get("unicode_col", "Segmented_Text"))
        LOGGER.info(
            "LocalModelsScorer initialised with %d models (use_unicode=%s)",
            len(self._ensemble), self._use_unicode,
        )

    def run(self) -> None:
        from sub_tasks.models_eval import bi_encoder_predict, reranker_predict

        LOGGER.info("Starting LocalModelsScorer run")

        input_file = self._files_manager.selected_pairs_current
        if not input_file.exists():
            raise FileNotFoundError(
                f"Selected pairs file not found: {input_file}"
            )

        pairs_df = load_dataframe(input_file)
        pairs_df.drop(columns=["cosine", "bin", "score"], inplace=True, errors="ignore")

        if self._use_unicode:
            needed = set(pairs_df["SentenceA"].astype(str)) | set(pairs_df["SentenceB"].astype(str))
            ewts_to_unicode = self._build_ewts_to_unicode_map(needed)
            text1 = [ewts_to_unicode.get(s, s) for s in pairs_df["SentenceA"].astype(str)]
            text2 = [ewts_to_unicode.get(s, s) for s in pairs_df["SentenceB"].astype(str)]
            LOGGER.info("LocalModelsScorer: using unicode text for scoring (%s col).", self._unicode_col)
        else:
            text1 = list(pairs_df["SentenceA"])
            text2 = list(pairs_df["SentenceB"])

        data = {"text1": text1, "text2": text2}

        all_norm_scores: list[np.ndarray] = []
        weights: list[float] = []

        for model_cfg in self._ensemble:
            name = model_cfg["name"]
            mtype = model_cfg["type"]
            weight = model_cfg.get("weight", 1.0 / len(self._ensemble))

            LOGGER.info("Scoring with %s (%s, weight=%.2f)", name, mtype, weight)

            model_batch_size = model_cfg.get("batch_size", 4)

            if mtype == "bi_encoder":
                raw = bi_encoder_predict(name, data, batch_size=model_batch_size)
            elif mtype == "reranker":
                instruction = model_cfg.get("instruction", "")
                raw = reranker_predict(
                    name, data, instruction=instruction, batch_size=model_batch_size
                )
            else:
                raise ValueError(f"Unknown model type: {mtype}")

            all_norm_scores.append(_normalize(raw))
            weights.append(weight)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        combined = sum(w * s for w, s in zip(weights, all_norm_scores))

        pairs_df["score"] = combined
        pairs_df["is_unicode_score"] = self._use_unicode

        output_file = self._files_manager.llms_pairs_scored_current
        save_dataframe_single(pairs_df, output_file, exists_ok=True)

        LOGGER.info(
            "LocalModelsScorer done – %d pairs scored -> %s",
            len(pairs_df),
            output_file,
        )

    def _build_ewts_to_unicode_map(self, needed: set) -> dict:
        """
        Scan the core dataset reading only the two text columns, chunk by chunk,
        and return a {ewts: unicode} dict for the requested sentences only.
        Stops early once all needed sentences are found.
        """
        core_path = str(self._files_manager.core_dataset)
        mapping: dict = {}
        remaining = set(needed)

        LOGGER.info(
            "Building EWTS→unicode map for %d unique sentences from %s",
            len(needed), core_path,
        )

        try:
            chunks = pd.read_csv(
                core_path,
                usecols=[self._ewts_col, self._unicode_col],
                chunksize=100_000,
                dtype=str,
            )
        except ValueError as e:
            raise KeyError(
                f"Core dataset must have '{self._ewts_col}' and '{self._unicode_col}' columns. "
                f"Error: {e}"
            )

        for chunk in chunks:
            hits = chunk[chunk[self._ewts_col].isin(remaining)]
            for ewts, uni in zip(hits[self._ewts_col], hits[self._unicode_col]):
                mapping[ewts] = uni
                remaining.discard(ewts)
            if not remaining:
                break  # all sentences resolved — no need to read further

        if remaining:
            LOGGER.warning(
                "%d sentences had no unicode match in the core dataset; EWTS text will be used as fallback.",
                len(remaining),
            )

        LOGGER.info("Built EWTS→unicode map with %d entries.", len(mapping))
        return mapping
