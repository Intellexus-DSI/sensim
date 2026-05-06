from __future__ import annotations

import random
import shutil
import os

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import logging
import yaml
import app_config
from common_utils import load_yaml


LOGGER = logging.getLogger(__name__)



class TrainingFilesManager:
    """
    Centralizes all filesystem paths used by the progressive learner pipeline.

    It creates a run folder (results_root/<run_datetime>) and per-iteration folders
    (it_00, it_01, ...), and exposes typed properties for each file produced/consumed.

    NOTE: Refactored for clarity and maintainability without changing path logic.
    """

    def __init__(
            self,
            data_dir: Optional[Union[str, Path]] = "data",
            models: Optional[List[str]] = None,
            *,
            # Run folder control
            results_root: Optional[Union[str, Path]] = None,
            run_datetime: Optional[Union[datetime, str]] = None,
            parent_dir: Optional[Union[str, Path]] = None,
            datetime_format: str = "%Y-%m-%d_%H-%M-%S",
            # Iterations
            start_iteration: Optional[int] = None,
            iteration_folder_base: str = "it",
            # Parent-level model dir
            model_dir: str = "ckpts/sts-b",
            # -------------------------
            # Result files (bases)
            # -------------------------
            selected_pairs_base: str = "selected_pairs",
            temp_ids_base: str = "temp_ids",
            sampled_4_pairs_base: str = "sampled_4_pairs",
            similarity_results_base: str = "similarity_results",
            model_similarity_results_base: str = "similarity_results",
            llms_4_pair_annotations_base: str = "llms_4_pair_annotations",
            formatted_llms_4_pair_annotations_base: Optional[str] = None,
            llms_pairs_scored_base: str = "llms_pairs_scored",
            evaluated_pairs_base: str = "evaluated_pairs",
            unlabeled_sampled_sentences_base: str = "unlabeled_sampled_sentences",
            embeddings_base: str = "embeddings",
            results_file_base: str = "active_sampling_results",
            # -------------------------
            # Data dir file bases
            # -------------------------
            validation_dataset: str | List[str] = "train_A_300_scored.xlsx",
            test_dataset: str | List[str] = None,
            segments_dataset: str = "sentences.xlsx",
    ) -> None:
        self.models: List[str] = models or []
        self.data_dir: Path = Path(data_dir)

        # Resolve base_dir from config
        _app_config = app_config.AppConfig()
        base_dir = Path(_app_config.get('sensim_base_dir', '.'))
        active_sampling_dir = Path(_app_config.get('active_sampling_dir', base_dir / "results" / "active_sampling"))

        # Folder base names
        if results_root is None:
            results_root = active_sampling_dir
        self.results_root: Path = Path(results_root)
        self.iteration_folder_base: str = iteration_folder_base
        self.model_dir_base: str = model_dir
        self.datetime_format: str = datetime_format

        # Result file bases
        self.selected_pairs_base: str = selected_pairs_base
        self.temp_ids_base: str = temp_ids_base
        self.sampled_4_pairs_base: str = sampled_4_pairs_base
        self.similarity_results_base: str = similarity_results_base
        self.model_similarity_results_base: str = model_similarity_results_base
        self.llms_4_pair_annotations_base: str = llms_4_pair_annotations_base
        self.formatted_llms_4_pair_annotations_base: str = (
                formatted_llms_4_pair_annotations_base or f"formatted_{self.llms_4_pair_annotations_base}"
        )
        self.llms_pairs_scored_base: str = llms_pairs_scored_base
        self.evaluated_pairs_base: str = evaluated_pairs_base
        self.unlabeled_sampled_sentences_base: str = unlabeled_sampled_sentences_base
        self.embeddings_base: str = embeddings_base
        self.sentences_base: str = "sentences"
        self.results_file_base: str = results_file_base
        # Data dir file bases
        self.core_dataset: Path = (self.data_dir / segments_dataset).expanduser().resolve()

        if test_dataset is None:
            test_dataset = ["train_B_300_scored.xlsx", "test_pairs_300_scored.xlsx", "validation_pairs_100_scored.xlsx"]

        if isinstance(validation_dataset, list):
            self.validation_dataset = [(self.data_dir / ds).expanduser().resolve() for ds in validation_dataset]
        else:
            self.validation_dataset = (self.data_dir / validation_dataset).expanduser().resolve()

        if isinstance(test_dataset, list):
            self.test_dataset = [(self.data_dir / ds).expanduser().resolve() for ds in test_dataset]
        else:
            self.test_dataset = (self.data_dir / test_dataset).expanduser().resolve()

        # Extensions
        self.xlsx_ext: str = ".xlsx"
        self.txt_ext: str = ".txt"
        self.csv_ext: str = ".csv"
        self.npy_ext: str = ".npy"

        # Resolve parent_dir
        if parent_dir is not None:
            self.parent_dir: Path = Path(parent_dir)
        else:
            dt = run_datetime or datetime.now()
            if isinstance(dt, str):
                dt_str = datetime.strptime(dt, self.datetime_format).strftime(self.datetime_format)
            elif isinstance(dt, datetime):
                dt_str = dt.strftime(self.datetime_format)
            else:
                raise ValueError("run_datetime must be either a str or datetime object")
            self.parent_dir = (self.results_root / dt_str).expanduser().resolve()

        # Iterations

        self.model_dir: Path = (self.parent_dir / self.model_dir_base).expanduser().resolve()

        self.parent_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Config handling: on resume use the run folder's config; on fresh run copy from project root
        run_config = self.parent_dir / app_config.CONFIG_FILE_NAME
        if run_datetime is not None and run_config.exists():
            # Resume: reload AppConfig from the run folder's config
            LOGGER.info("Resume mode: reloading config from %s", run_config)
            _app_config.reload(str(run_config))
        else:
            # Fresh run (or first time): copy project-root config into the run folder
            config_source = Path(app_config.CONFIG_FILE_NAME)
            shutil.copy2(config_source, self.parent_dir)
            LOGGER.info("Copied %s -> %s", config_source, run_config)

            # If seed is absent or None in the copied config, generate one and persist it
            _cfg = load_yaml(str(run_config))
            if _cfg.get("seed") is None:
                _cfg["seed"] = random.randint(0, 2 ** 31 - 1)
                with open(run_config, "w", encoding="utf-8") as _f:
                    yaml.safe_dump(_cfg, _f, allow_unicode=True, sort_keys=False)
                LOGGER.info("seed was None — randomized and saved seed=%d to %s", _cfg["seed"], run_config)

            _app_config.reload(str(run_config))

        # Apply no_train_eval mode from the (now reloaded) config
        no_train_eval = _app_config.get("no_train_eval", False)
        if no_train_eval:
            LOGGER.info("no_train_eval mode: using all_gold_pairs_1000_scored.xlsx as test, no validation.")
            self.validation_dataset = []
            self.test_dataset = [(self.data_dir / "all_gold_pairs_1000_scored.xlsx").expanduser().resolve()]

        self.current_iteration: int = self._resolve_start_iteration(start_iteration)
        self.previous_iteration: Optional[int] = self.current_iteration - 1 if self.current_iteration > 0 else None
        self._ensure_iteration_dir(self.current_iteration)

        LOGGER.info(
            f"TrainingFilesManager initialized. Parent dir: {self.parent_dir}, Current iteration: {self.current_iteration}")

        # -----------------------------

    # Iteration folders
    # -----------------------------

    def _iter_dir_name(self, i: int) -> str:
        return f"{self.iteration_folder_base}_{i:02d}"

    def iteration_dir(self, i: int) -> Path:
        if i < 0:
            raise ValueError("Iteration index must be >= 0")
        return (self.parent_dir / self._iter_dir_name(i)).expanduser().resolve()

    def _find_existing_iterations(self) -> List[int]:
        """
        Scan parent_dir for folders matching: {iteration_folder_base}_NN
        Returns sorted list of ints.
        """
        if not self.parent_dir.exists():
            return []

        prefix = f"{self.iteration_folder_base}_"
        out: List[int] = []

        for p in self.parent_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            # allow exactly 2 digits like 00..99 (matches your format)
            if len(suffix) != 2 or not suffix.isdigit():
                continue
            out.append(int(suffix))

        out.sort()
        return out

    def _resolve_start_iteration(self, start_iteration: Optional[int]) -> int:
        """
        If start_iteration is None:
          - If iteration folders exist → use max existing
          - Else → 0
        Else:
          - validate >= 0 and use it
        """
        if start_iteration is None:
            existing = self._find_existing_iterations()
            if existing:
                last_it = existing[-1]
                return last_it
            return 0

        if start_iteration < 0:
            raise ValueError("start_iteration must be >= 0 or None")
        return start_iteration

    @property
    def current_iteration_dir(self) -> Path:
        return self.iteration_dir(self.current_iteration)

    @property
    def previous_iteration_dir(self) -> Optional[Path]:
        return self.iteration_dir(self.previous_iteration) if self.previous_iteration is not None else None

    def _ensure_iteration_dir(self, i: int) -> None:
        self.iteration_dir(i).mkdir(parents=True, exist_ok=True)

    def increment(self) -> int:
        """Advance to the next iteration and ensure its directory exists."""
        self.previous_iteration = self.current_iteration
        self.current_iteration += 1
        self._ensure_iteration_dir(self.current_iteration)
        return self.current_iteration

    # -----------------------------
    # Parent-level dirs/files
    # -----------------------------

    @property
    def main_results_path(self) -> Path:
        folder_name = self.parent_dir.name
        return (self.parent_dir / f"{self.results_file_base}_{folder_name}{self.xlsx_ext}").expanduser().resolve()

    @property
    def results_parent_dir(self) -> Path:
        return self.parent_dir

    # -----------------------------
    # Internal file path helpers
    # -----------------------------

    def _iter_xlsx(self, base: str, i: int) -> Path:
        return (self.iteration_dir(i) / f"{base}_{i:02d}{self.xlsx_ext}").expanduser().resolve()

    def _iter_csv(self, base: str, i: int) -> Path:
        return (self.iteration_dir(i) / f"{base}_{i:02d}{self.csv_ext}").expanduser().resolve()

    def _iter_txt(self, base: str, i: int) -> Path:
        return (self.iteration_dir(i) / f"{base}_{i:02d}{self.txt_ext}").expanduser().resolve()

    def _iter_npy(self, base: str, i: int) -> Path:
        return (self.iteration_dir(i) / f"{base}_{i:02d}{self.npy_ext}").expanduser().resolve()

    def _current_xlsx(self, base: str) -> Path:
        return self._iter_xlsx(base, self.current_iteration)

    def _previous_xlsx(self, base: str) -> Optional[Path]:
        return self._iter_xlsx(base, self.previous_iteration) if self.previous_iteration is not None else None

    def _current_csv(self, base: str) -> Path:
        return self._iter_csv(base, self.current_iteration)

    def _previous_csv(self, base: str) -> Optional[Path]:
        return self._iter_csv(base, self.previous_iteration) if self.previous_iteration is not None else None

    def _current_txt(self, base: str) -> Path:
        return self._iter_txt(base, self.current_iteration)

    def _previous_txt(self, base: str) -> Optional[Path]:
        return self._iter_txt(base, self.previous_iteration) if self.previous_iteration is not None else None

    def _current_npy(self, base: str) -> Path:
        return self._iter_npy(base, self.current_iteration)

    def _previous_npy(self, base: str) -> Optional[Path]:
        return self._iter_npy(base, self.previous_iteration) if self.previous_iteration is not None else None

    def _validate_model(self, model: str) -> None:
        if self.models and model not in self.models:
            raise ValueError(
                f"Model '{model}' is not in allowed models list: {self.models}. "
                "Pass an empty models list to disable validation."
            )

    # -----------------------------
    # selected_pairs
    # -----------------------------
    def selected_pairs_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.selected_pairs_base, i)

    @property
    def selected_pairs_current(self) -> Path:
        return self._current_xlsx(self.selected_pairs_base)

    @property
    def selected_pairs_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.selected_pairs_base)

    # -----------------------------
    # evaluated_pairs
    # -----------------------------
    def evaluated_pairs_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.evaluated_pairs_base, i)

    @property
    def evaluated_pairs_current(self) -> Path:
        return self._current_xlsx(self.evaluated_pairs_base)

    @property
    def evaluated_pairs_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.evaluated_pairs_base)

    # -----------------------------
    # unlabeled_sampled_sentences
    # -----------------------------
    def unlabeled_sampled_sentences_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.unlabeled_sampled_sentences_base, i)

    @property
    def unlabeled_sampled_sentences_current(self) -> Path:
        return self._current_xlsx(self.unlabeled_sampled_sentences_base)

    @property
    def unlabeled_sampled_sentences_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.unlabeled_sampled_sentences_base)

    # -----------------------------
    # temp_ids
    # -----------------------------
    def temp_ids_iteration(self, i: int) -> Path:
        return self._iter_txt(self.temp_ids_base, i)

    @property
    def temp_ids_current(self) -> Path:
        return self._current_txt(self.temp_ids_base)

    @property
    def temp_ids_previous(self) -> Optional[Path]:
        return self._previous_txt(self.temp_ids_base)

    # -----------------------------
    # sampled_4_pairs
    # -----------------------------
    def sampled_4_pairs_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.sampled_4_pairs_base, i)

    @property
    def sampled_4_pairs_current(self) -> Path:
        return self._current_xlsx(self.sampled_4_pairs_base)

    @property
    def sampled_4_pairs_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.sampled_4_pairs_base)

    # -----------------------------
    # model-specific sampled_4_pairs
    # -----------------------------
    def model_sampled_4_pairs_current(self, model: str) -> Path:
        self._validate_model(model)
        base = f"{model}_{self.sampled_4_pairs_base}"
        return self._current_xlsx(base)

    # -----------------------------
    # model-specific similarity_results
    # -----------------------------
    def model_similarity_results_iteration(self, model: str, i: int) -> Path:
        self._validate_model(model)
        base = f"{model}_{self.model_similarity_results_base}"
        return self._iter_xlsx(base, i)

    def model_similarity_results_current(self, model: str) -> Path:
        self._validate_model(model)
        base = f"{model}_{self.model_similarity_results_base}"
        return self._current_xlsx(base)

    def model_similarity_results_previous(self, model: str) -> Optional[Path]:
        self._validate_model(model)
        base = f"{model}_{self.model_similarity_results_base}"
        return self._previous_xlsx(base)

    # -----------------------------
    # llms_4_pair_annotations
    # -----------------------------
    def llms_4_pair_annotations_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.llms_4_pair_annotations_base, i)

    @property
    def llms_4_pair_annotations_current(self) -> Path:
        return self._current_xlsx(self.llms_4_pair_annotations_base)

    @property
    def llms_4_pair_annotations_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.llms_4_pair_annotations_base)

    # -----------------------------
    # formatted_llms_4_pair_annotations
    # -----------------------------
    def formatted_llms_4_pair_annotations_iteration(self, i: int) -> Path:
        return self._iter_csv(self.formatted_llms_4_pair_annotations_base, i)

    @property
    def formatted_llms_4_pair_annotations_current(self) -> Path:
        return self._current_csv(self.formatted_llms_4_pair_annotations_base)

    @property
    def formatted_llms_4_pair_annotations_previous(self) -> Optional[Path]:
        return self._previous_csv(self.formatted_llms_4_pair_annotations_base)

    # -----------------------------
    # llms_pairs_scored
    # -----------------------------
    def llms_pairs_scored_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.llms_pairs_scored_base, i)

    @property
    def llms_pairs_scored_current(self) -> Path:
        return self._current_xlsx(self.llms_pairs_scored_base)

    @property
    def llms_pairs_scored_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.llms_pairs_scored_base)

    # -----------------------------
    # Merged trainset (cumulative across iterations)
    # -----------------------------
    @property
    def merged_trainset_path(self) -> Path:
        folder_name = self.parent_dir.name
        return (self.parent_dir / f"merged_trainset_{folder_name}{self.xlsx_ext}").expanduser().resolve()

    def build_merged_trainset(self) -> Path:
        """Merge all llms_pairs_scored files from iteration 1 up to current_iteration into a single file."""
        import pandas as pd
        dfs = []
        for i in range(1, self.current_iteration + 1):
            path = self.llms_pairs_scored_iteration(i)
            if path.exists():
                df = pd.read_excel(path)
                df['iteration_source'] = i
                dfs.append(df)
                LOGGER.info("Merged trainset: added %d rows from %s", len(df), path.name)
            else:
                LOGGER.warning("Merged trainset: missing %s, skipping.", path)

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged.to_excel(self.merged_trainset_path, index=False)
            LOGGER.info("Merged trainset: %d total rows -> %s", len(merged), self.merged_trainset_path)
        else:
            LOGGER.warning("No scored files found to merge.")
        return self.merged_trainset_path

    # -----------------------------
    # embeddings
    # -----------------------------
    def embeddings_iteration(self, i: int) -> Path:
        return self._iter_npy(self.embeddings_base, i)

    @property
    def embeddings_current(self) -> Path:
        return self._current_npy(self.embeddings_base)

    @property
    def embeddings_previous(self) -> Optional[Path]:
        return self._previous_npy(self.embeddings_base)

    # -----------------------------
    # sentences
    # -----------------------------
    def sentences_iteration(self, i: int) -> Path:
        return self._iter_npy(self.sentences_base, i)

    @property
    def sentences_current(self) -> Path:
        return self._current_npy(self.sentences_base)

    @property
    def sentences_previous(self) -> Optional[Path]:
        return self._previous_npy(self.sentences_base)

    # -----------------------------
    # Results file (iteration-scoped)
    # -----------------------------
    def results_file_iteration(self, i: int) -> Path:
        return self._iter_xlsx(self.results_file_base, i)

    @property
    def results_file_current(self) -> Path:
        return self._current_xlsx(self.results_file_base)

    @property
    def results_file_previous(self) -> Optional[Path]:
        return self._previous_xlsx(self.results_file_base)
