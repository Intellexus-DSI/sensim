from __future__ import annotations

import logging
import random
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from app_config import AppConfig
from common_utils import load_dataframe, save_dataframe_single, load_yaml
from common_utils import resolve_path
from training_files_manager import TrainingFilesManager
from .faiss_index import build_faiss_index, ensure_2d_float32
from .utils import calculate_wylie_distance

app_config = AppConfig()

LOGGER = logging.getLogger(__name__)


@dataclass
class SimPairsSampler:
    """
    Samples k unique sentences and pairs each one with its d-th nearest neighbor
    using FAISS on GPU.

    - Embeddings are loaded from files_manager.embeddings_previous()
    - Sentences are loaded from files_manager.core_dataset, using unique values from `sent_col`
    - Similarity: cosine (implemented as inner product on L2-normalized vectors)
    - Output is saved to files_manager.selected_pairs_current (or selected_current if exists)
      with columns ["ID", "SentenceA", "SentenceB"]
    """
    pairs_per_batch: int
    search_k: int
    minimum_distance: float
    maximum_distance: float
    max_iterations_for_distance_filtering: int
    use_fixed_min_distance: bool
    sentence_col: str
    embeddings_col: str
    seed: int

    # IO
    _files_manager: TrainingFilesManager

    DEFAULTS = {
        "pairs_per_batch": 250,
        "search_k": 20,
        "minimum_distance": 0.880,
        "maximum_distance": 0.980,
        "max_iterations_for_distance_filtering": 11,
        "use_fixed_min_distance": False,
        "seed": 42,
        "sentence_col": "Segmented_Text_EWTS",
        "embeddings_col": "embeddings",
        "min_syllable_distance": 5,
        "beta_distribution_a": None,
        "beta_distribution_b": None,
    }

    def __init__(self,
                 # Files manager
                 files_manager: TrainingFilesManager,

                 # Config / keys
                 config_path: Optional[str] = None,

                 # Sampler params
                 pairs_per_batch: Optional[int] = None,
                 search_k: Optional[int] = None,
                 minimum_distance: Optional[float] = None,
                 maximum_distance: Optional[float] = None,
                 max_iterations_for_distance_filtering: Optional[int] = None,
                 use_fixed_min_distance: Optional[bool] = None,

                 sentence_col: Optional[str] = None,
                 embeddings_col: Optional[str] = None,
                 min_syllable_distance: Optional[int] = None,
                 beta_distribution_a: Optional[float] = None,
                 beta_distribution_b: Optional[float] = None,

                 # Misc
                 seed: Optional[int] = None,

                 # Command-line arguments
                 argv: Optional[List[str]] = None,

                 # Overrides
                 **overrides: Any,
                 ) -> None:
        cli_args, maybe_config_path = self._parse_cli(argv)
        yaml_args = load_yaml(config_path or maybe_config_path)

        def choose(key: str, given: Any = None, *aliases: str) -> Any:
            """Select value in order: overrides > given_arg > CLI > YAML > aliases > default."""
            # explicit argument
            if given is not None:
                return given
            # overrides
            v = overrides.get(key, None)
            if v is not None:
                return v
            # CLI
            v = cli_args.get(key, None)
            if v is not None:
                return v
            # YAML (primary key then aliases)
            v = yaml_args.get(key, None)
            if v is not None:
                return v
            for alias in aliases:
                v = yaml_args.get(alias, None)
                if v is not None:
                    return v
            # default
            return self.DEFAULTS.get(key, None)

        self._files_manager = files_manager
        self.pairs_per_batch = choose("pairs_per_batch", pairs_per_batch)
        self.search_k = choose("search_k", search_k)
        self.minimum_distance = choose("minimum_distance", minimum_distance, "calc_min_distance_end_val")
        self.maximum_distance = choose("maximum_distance", maximum_distance, "calc_min_distance_minimum_distance")
        self.max_iterations_for_distance_filtering = choose("max_iterations_for_distance_filtering",
                                                            max_iterations_for_distance_filtering,
                                                            "calc_min_distance_max_iterations")
        self.use_fixed_min_distance = bool(choose("use_fixed_min_distance", use_fixed_min_distance))
        self.seed = choose("seed", seed)
        self.sentence_col = choose("sentence_col", sentence_col)
        self.embeddings_col = choose("embeddings_col", embeddings_col)
        self.min_syllable_distance = choose("min_syllable_distance", min_syllable_distance)
        self.beta_distribution_a = choose("beta_distribution_a", beta_distribution_a)
        self.beta_distribution_b = choose("beta_distribution_b", beta_distribution_b)

        self.pair_dist_generator = np.random.default_rng(self.seed)
        self.samples_generator = np.random.default_rng(self.seed)
        self.last_batch_count = None  # populated after each run
        self.last_faiss_distance_mean = None
        self.last_faiss_distance_std = None
        self.last_min_dist = None

    # -------------------------
    # Public API
    # -------------------------
    def run(self) -> None:
        """
        Main entry point:
        1) Load sentences and embeddings
        2) Build FAISS index on GPU
        3) For each of k sampled sentences, find its d-th nearest neighbor (excluding self) that meets the distance threshold, and save the pair.
         - To find the d-th neighbor, we search for the top (d + extra) neighbors and randomly pick one of the top d (excluding self) to add some randomness and avoid duplicates.
         - We also apply a minimum distance threshold that decays over iterations to ensure we get more distant pairs in early iterations and can relax it in later iterations as the pool shrinks.
         - Additionally, we filter pairs based on a syllable distance (using calculate_wylie_distance) to ensure linguistic diversity, only keeping pairs with a syllable distance greater than 5.
         - We continue sampling until we have k valid pairs or exhaust the pool of sentences.
         - The output is saved to a file with columns ["ID", "SentenceA", "SentenceB", "FaissDistance"] for traceability and potential debugging.
         - The ID format includes the iteration number and a unique index to ensure uniqueness across iterations.
        """
        LOGGER.info("FAISS sampler: starting (k=%d, seed=%d)", self.pairs_per_batch, self.seed)

        core_sentences = self._load_core_sentences()
        embeddings, emb_sentences = self._load_embeddings()

        # Choose which sentence list we’ll use to index:
        # - If embeddings file includes sentences: trust it (alignment guaranteed).
        # - Else: assume embeddings are aligned to core unique sentences by index order.
        if emb_sentences is not None:
            sentences = emb_sentences
        else:
            sentences = core_sentences

        self._validate_inputs(embeddings=embeddings, sentences=sentences)

        # Build GPU index
        LOGGER.info(f"Starting FAISS indexing of embeddings (shape=%d)", embeddings.shape[0])
        faiss_index = build_faiss_index(embeddings)
        LOGGER.info(f"Finished FAISS indexing of embeddings (shape=%d)", embeddings.shape[0])

        pairs_df = self._build_min_dist_pairs_dataframe(
            faiss_index=faiss_index,
            embeddings=embeddings,
            sentences=sentences,
        )

        save_dataframe_single(pairs_df, self._files_manager.selected_pairs_current)
        self.seed += 1
        LOGGER.info("FAISS sampler: saved %d pairs -> %s", len(pairs_df), self._files_manager.selected_pairs_current)

        # In next run we will build a new index, so we can free memory from the current one right away.
        del faiss_index

    # -------------------------
    # Loading
    # -------------------------
    def _load_core_sentences(self) -> List[str]:
        df = load_dataframe(self._files_manager.core_dataset)
        if self.sentence_col not in df.columns:
            raise KeyError(
                f"core_dataset {self._files_manager.core_dataset} missing column '{self.sentence_col}'. Available: {list(df.columns)}")

        sentences = df[self.sentence_col].dropna().drop_duplicates().astype(str).tolist()
        LOGGER.info("Loaded %d unique core sentences from %s", len(sentences), self._files_manager.core_dataset)
        return sentences

    def _load_embeddings(self) -> Tuple[np.ndarray, Optional[List[str]]]:
        emb_path = self._files_manager.embeddings_previous
        if emb_path is None:
            raise FileNotFoundError("embeddings_previous() is None (no previous iteration).")

        emb_path = resolve_path(emb_path, must_exist=True, allowed_suffixes=[".npy"])

        LOGGER.info("Loading embeddings from %s", emb_path)
        arr = np.load(emb_path).astype(np.float32, copy=False)

        sentences = None
        sent_path = self._files_manager.sentences_previous
        if sent_path is not None and sent_path.exists():
            LOGGER.info("Loading sentences from %s", sent_path)
            sentences = np.load(sent_path, allow_pickle=True).tolist()

        return ensure_2d_float32(arr), sentences

    # -------------------------
    # Validation / utils
    # -------------------------
    def _validate_inputs(self, *, embeddings: np.ndarray, sentences: List[str]) -> None:
        if len(sentences) < self.pairs_per_batch:
            raise ValueError(f"Not enough sentences to sample: have {len(sentences)}, need k={self.pairs_per_batch}")
        if embeddings.shape[0] != len(sentences):
            raise ValueError(
                f"Embeddings count != sentences count: embeddings={embeddings.shape[0]}, sentences={len(sentences)}. "
                "Your embeddings file must align to the sentence list used for indexing."
            )

    # -------------------------
    # Pair building
    # -------------------------
    def _build_min_dist_pairs_dataframe(
            self,
            *,
            faiss_index,
            embeddings: np.ndarray,
            sentences: List[str],
    ) -> pd.DataFrame:

        minimum_distance = self._calc_min_distance()
        self.last_min_dist = minimum_distance
        LOGGER.info(
            f'Iteration {self._files_manager.current_iteration}. Estimated minimum FAISS distance for d-th neighbor: {minimum_distance:.6f}')

        pairs = self._find_distant_pairs_from_stream(faiss_index, embeddings, sentences, minimum_distance)
        if len(pairs) < self.pairs_per_batch:
            LOGGER.warning("Only found %d pairs with min distance %.6f (requested %d)", len(pairs), minimum_distance,
                           self.pairs_per_batch)

        df = pd.DataFrame(pairs, columns=["ID", "SentenceA", "SentenceB", "FaissDistance"])

        if len(df) > 0:
            self.last_faiss_distance_mean = float(df["FaissDistance"].mean())
            self.last_faiss_distance_std = float(df["FaissDistance"].std())
            LOGGER.info("FaissDistance stats: mean=%.6f, std=%.6f",
                        self.last_faiss_distance_mean, self.last_faiss_distance_std)

        return df

    def _calc_min_distance(self) -> float:
        if self.use_fixed_min_distance:
            return self.maximum_distance
        b = (self.maximum_distance - self.minimum_distance) / np.log(
            self.max_iterations_for_distance_filtering)
        y = self.maximum_distance - b * np.log(self._files_manager.current_iteration)
        return y

    def _get_minimum_distance(self, faiss_index, embeddings: np.ndarray, n_samples, top_k) -> float:

        # Sample n_samples unique indexes
        length = embeddings.shape[0]
        sampled_indexes = self.samples_generator.choice(length, size=n_samples, replace=False)
        selected_embeddings = embeddings[sampled_indexes]
        nn_distances, nn_ids = faiss_index.search(selected_embeddings, 1)

        # nn_distances shape: (num_queries, k)
        # We only care about the first column (distance to the very nearest neighbor)
        first_distances = nn_distances[:, 0]

        # 1. Sort: Get indices that would sort the array from Low -> High
        sorted_indices = np.argsort(first_distances)

        # 2. Reverse: We want High -> Low (Highest distances first)
        sorted_indices_desc = sorted_indices[::-1]

        # 3. Cut: Take the top x indices
        top_x_indices = sorted_indices_desc[:top_k]

        # 4. Take the values: Get the actual distances for these queries
        top_x_values = first_distances[top_x_indices]

        average_distance = top_x_values.mean()

        return average_distance

    def _find_distant_pairs_from_stream(self, faiss_index, embeddings, sentences: List[str], min_dist):
        """
        Consumes indices from a generator, fetches their embeddings, and finds distant pairs.

        Args:
            faiss_index: Trained FAISS index
            embeddings: np.ndarray of shape (total_count, dim)
            sentences: List of sentences corresponding to embeddings
            min_dist: Minimum distance threshold

        Returns:
            list of [query_idx, neighbor_idx, distance]
        """
        t_start = time.perf_counter()
        batch_count = 0
        collected_pairs = []
        index_stream = self._unique_random_stream(len(sentences) - 1, self.seed)

        pbar = tqdm(total=self.pairs_per_batch, desc="Collecting pairs. Matches counter:0")
        while len(collected_pairs) < self.pairs_per_batch:
            batch_count += 1
            pbar.set_description(f"Collecting pairs. Matches counter:{batch_count * self.pairs_per_batch}")

            batch_min_dist = min_dist
            if self.beta_distribution_a and self.beta_distribution_b:
                distance_range = self.maximum_distance - min_dist
                beta_distribution = self.samples_generator.beta(self.beta_distribution_a, self.beta_distribution_b)
                batch_min_dist = self.maximum_distance - (beta_distribution * distance_range)


            if batch_count % 25 == 0:
                LOGGER.info(f'Iteration {self._files_manager.current_iteration}, batch {batch_count}. batch_min_dist: {batch_min_dist:.6f}. Collected pairs so far: {len(collected_pairs)}')

            # 1. Consume a batch of indices from the stream
            # islice pulls the next 'pairs_per_batch' items from the generator
            batch_indices = list(islice(index_stream, self.pairs_per_batch))

            # If stream is empty, stop
            if not batch_indices:
                LOGGER.warning(f'Stream exhausted after {batch_count} batches! Collected pairs: {len(collected_pairs)}')
                break

            # 2. Fetch the actual vectors for these indices
            # We use fancy indexing to get the specific rows
            batch_queries = embeddings[batch_indices]

            # 3. Search in FAISS
            D, I = faiss_index.search(batch_queries, self.search_k)

            # 3.5 Pre-filter: keep only queries that have at least one valid neighbor
            batch_indices_arr = np.array(batch_indices)
            valid_mask = (D >= batch_min_dist) & (I != batch_indices_arr[:, np.newaxis])
            has_valid = valid_mask.any(axis=1)
            batch_indices = [idx for idx, v in zip(batch_indices, has_valid) if v]
            D = D[has_valid]
            I = I[has_valid]

            # 4. Filter results
            for local_i, query_idx in enumerate(batch_indices):
                # Iterate through neighbors (k)
                # for k in range(search_k):
                for k in self.samples_generator.permutation(self.search_k):
                    dist = D[local_i, k]
                    neighbor_idx = I[local_i, k]

                    # Filter: Distance check & Self-match check
                    # (We usually want to skip if query == neighbor, assuming dist approx 0)
                    if dist >= batch_min_dist and query_idx != neighbor_idx:
                        sentence_a = sentences[int(query_idx)]
                        sentence_b = sentences[int(neighbor_idx)]
                        syllable_dist = calculate_wylie_distance(sentence_a, sentence_b)
                        if syllable_dist > self.min_syllable_distance:
                            collected_pairs.append({
                                "ID": f"pair_{self._files_manager.current_iteration:02d}_{batch_count:03d}_{local_i:03d}_{k:02d}",
                                "SentenceA": sentence_a,
                                "SentenceB": sentence_b,
                                "FaissDistance": dist,
                            })
                            pbar.update(1)

                            # proceed to next neighbor
                            #continue
                            break

                # Check global stop condition inside the loop
                if len(collected_pairs) >= self.pairs_per_batch:
                    break

            # Also check global stop condition at the end of the batch, in case we collected enough pairs in this batch
            if len(collected_pairs) >= self.pairs_per_batch:
                break

        pbar.close()
        elapsed = time.perf_counter() - t_start
        self.last_batch_count = batch_count
        LOGGER.info(
            "_find_distant_pairs_from_stream finished in %.2fs — %d batches, %d pairs collected",
            elapsed, batch_count, len(collected_pairs),
        )
        return collected_pairs[:self.pairs_per_batch]

    @staticmethod
    def _unique_random_stream(n, seed):
        """Yields unique random numbers from 0 to n (inclusive) with a seed."""
        # Create a local random instance
        rng = random.Random(seed)

        # Create the pool and shuffle it
        pool = list(range(n + 1))
        rng.shuffle(pool)

        # Yield one by one
        yield from pool

    @staticmethod
    def _parse_cli(argv: Optional[List[str]]) -> tuple[Dict[str, Any], Optional[str]]:
        if argv is None:
            return {}, None
        p = ArgumentParser(
            description="FAISS-based Pair Sampler",
            allow_abbrev=False, )

        # Config / keys
        p.add_argument("--config", type=str, help="Path to config YAML")

        p = ArgumentParser(add_help=False)

        # Sampler params
        p.add_argument("--pairs-per-batch", type=int, help="Number of unique sentences to sample.")
        p.add_argument("--search-k", type=int, help="Number of nearest neighbors to search for each sentence.")
        p.add_argument("--minimum-distance", type=float, help="Minimum distance for candidate pairs.")
        p.add_argument("--maximum-distance", type=float, help="Maximum distance for candidate pairs.")
        p.add_argument("--max-iterations-for-distance-filtering", type=int,
                       help="Maximum iterations limit for distance filtering.")
        p.add_argument("--sentence-col", type=str, help="Column name in core_dataset containing sentences.")
        p.add_argument("--embeddings-col", type=str, help="Column name in embeddings file containing embeddings.")
        # Misc
        p.add_argument("--seed", type=int, help="Random seed")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)

        maybe_config_path = raw.pop("config", None)

        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
