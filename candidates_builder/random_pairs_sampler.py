from __future__ import annotations

import logging
import time
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app_config import AppConfig
from common_utils import load_dataframe, save_dataframe_single, load_yaml
from training_files_manager import TrainingFilesManager
from .utils import calculate_wylie_distance

app_config = AppConfig()

LOGGER = logging.getLogger(__name__)


class RandomPairsSampler:
    """
    Samples pairs of sentences completely at random from the core dataset.

    No FAISS, no distance filtering — pure random selection.
    Applies only the syllable-distance filter to ensure basic linguistic diversity.
    Output columns match SimPairsSampler: ["ID", "SentenceA", "SentenceB", "FaissDistance"]
    with FaissDistance set to NaN.
    """

    DEFAULTS = {
        "pairs_per_batch": 250,
        "seed": 42,
        "sentence_col": "Segmented_Text_EWTS",
        "min_syllable_distance": 5,
    }

    def __init__(
        self,
        files_manager: TrainingFilesManager,
        config_path: Optional[str] = None,
        pairs_per_batch: Optional[int] = None,
        sentence_col: Optional[str] = None,
        min_syllable_distance: Optional[int] = None,
        seed: Optional[int] = None,
        argv: Optional[List[str]] = None,
        **overrides: Any,
    ) -> None:
        cli_args, maybe_config_path = self._parse_cli(argv)
        yaml_args = load_yaml(config_path or maybe_config_path)

        def choose(key: str, given: Any = None) -> Any:
            if given is not None:
                return given
            v = overrides.get(key)
            if v is not None:
                return v
            v = cli_args.get(key)
            if v is not None:
                return v
            v = yaml_args.get(key)
            if v is not None:
                return v
            return self.DEFAULTS.get(key)

        self._files_manager = files_manager
        self.pairs_per_batch = int(choose("pairs_per_batch", pairs_per_batch))
        self.sentence_col = str(choose("sentence_col", sentence_col))
        self.min_syllable_distance = int(choose("min_syllable_distance", min_syllable_distance))
        self.seed = int(choose("seed", seed))

        # Compatibility attributes (populated after run())
        self.last_batch_count: Optional[int] = None
        self.last_faiss_distance_mean: Optional[float] = None
        self.last_faiss_distance_std: Optional[float] = None
        self.last_min_dist: Optional[float] = None

    # -------------------------
    # Public API
    # -------------------------
    def run(self) -> None:
        LOGGER.info(
            "RandomPairsSampler: starting (k=%d, seed=%d, min_syllable_dist=%d)",
            self.pairs_per_batch, self.seed, self.min_syllable_distance,
        )
        t_start = time.perf_counter()

        sentences = self._load_core_sentences()
        pairs = self._sample_random_pairs(sentences)

        df = pd.DataFrame(pairs, columns=["ID", "SentenceA", "SentenceB", "FaissDistance"])
        save_dataframe_single(df, self._files_manager.selected_pairs_current)

        elapsed = time.perf_counter() - t_start
        self.last_batch_count = 1
        LOGGER.info(
            "RandomPairsSampler: saved %d pairs -> %s (%.2fs)",
            len(df), self._files_manager.selected_pairs_current, elapsed,
        )
        self.seed += 1

    # -------------------------
    # Core logic
    # -------------------------
    def _load_core_sentences(self) -> List[str]:
        df = load_dataframe(self._files_manager.core_dataset)
        if self.sentence_col not in df.columns:
            raise KeyError(
                f"core_dataset missing column '{self.sentence_col}'. "
                f"Available: {list(df.columns)}"
            )
        sentences = df[self.sentence_col].dropna().drop_duplicates().astype(str).tolist()
        LOGGER.info("Loaded %d unique core sentences.", len(sentences))
        return sentences

    def _sample_random_pairs(self, sentences: List[str]) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        n = len(sentences)
        if n < 2:
            raise ValueError(f"Need at least 2 sentences, got {n}.")

        collected: List[Dict[str, Any]] = []
        seen: set = set()
        max_attempts = self.pairs_per_batch * 20

        for attempt in range(max_attempts):
            if len(collected) >= self.pairs_per_batch:
                break

            a, b = rng.choice(n, size=2, replace=False)
            key = (int(min(a, b)), int(max(a, b)))
            if key in seen:
                continue
            seen.add(key)

            sentence_a = sentences[int(a)]
            sentence_b = sentences[int(b)]

            if calculate_wylie_distance(sentence_a, sentence_b) <= self.min_syllable_distance:
                continue

            iteration = self._files_manager.current_iteration
            collected.append({
                "ID": f"pair_{iteration:02d}_rand_{len(collected):05d}",
                "SentenceA": sentence_a,
                "SentenceB": sentence_b,
                "FaissDistance": float("nan"),
            })

        if len(collected) < self.pairs_per_batch:
            LOGGER.warning(
                "Only collected %d / %d pairs after %d attempts.",
                len(collected), self.pairs_per_batch, max_attempts,
            )
        return collected

    # -------------------------
    # CLI
    # -------------------------
    @staticmethod
    def _parse_cli(argv: Optional[List[str]]):
        if argv is None:
            return {}, None
        p = ArgumentParser(description="Random Pair Sampler", add_help=False)
        p.add_argument("--config", type=str)
        p.add_argument("--pairs-per-batch", type=int)
        p.add_argument("--sentence-col", type=str)
        p.add_argument("--min-syllable-distance", type=int)
        p.add_argument("--seed", type=int)
        args, _ = p.parse_known_args(argv)
        raw = vars(args)
        maybe_config_path = raw.pop("config", None)
        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
