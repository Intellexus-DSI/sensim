from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import paired_cosine_distances

from training_files_manager import TrainingFilesManager
from common_utils import load_dataframe, save_dataframe_single, load_yaml
from common_utils import clean_sentences_df, ensure_sentence_id_column
from candidates_builder.binning import assign_minmax_bins, select_bins_distributed

LOGGER = logging.getLogger(__name__)


@dataclass
class PairsSampler:
    # clustering + sampling params
    n_clusters: int
    pairs_per_cluster_pair: int
    k_selected: int
    bins: int

    sentence_col: str
    embeddings_col: str
    seed: int

    _files_manager: TrainingFilesManager

    DEFAULTS = {
        "n_clusters": 10,
        "pairs_per_cluster_pair": 4000,
        "k_selected": 250,
        "bins": 10,
        "seed": 42,
        "sentence_col": "Segmented_Text_EWTS",
        "embeddings_col": "embeddings",
    }

    def __init__(
            self,
            files_manager: TrainingFilesManager,
            config_path: Optional[str] = None,
            n_clusters: Optional[int] = None,
            pairs_per_cluster_pair: Optional[int] = None,
            k_selected: Optional[int] = None,
            bins: Optional[int] = None,
            sentence_col: Optional[str] = None,
            embeddings_col: Optional[str] = None,
            seed: Optional[int] = None,
            argv: Optional[List[str]] = None,
            **overrides: Any,
    ) -> None:
        cli_args, maybe_config_path = self._parse_cli(argv)
        yaml_args = load_yaml(config_path or maybe_config_path)

        def choose(key: str, given: Any = None) -> Any:
            if given is not None:
                return given
            v = overrides.get(key, None)
            if v is not None:
                return v
            v = cli_args.get(key, None)
            if v is not None:
                return v
            v = yaml_args.get(key, None)
            if v is not None:
                return v
            return self.DEFAULTS.get(key, None)

        self._files_manager = files_manager
        self.n_clusters = int(choose("n_clusters", n_clusters))
        self.pairs_per_cluster_pair = int(choose("pairs_per_cluster_pair", pairs_per_cluster_pair))
        self.k_selected = int(choose("k_selected", k_selected))
        self.bins = int(choose("bins", bins))
        self.seed = int(choose("seed", seed)) + self._files_manager.current_iteration
        self.sentence_col = str(choose("sentence_col", sentence_col))
        self.embeddings_col = str(choose("embeddings_col", embeddings_col))

        self._validate_args()

    def _validate_args(self) -> None:
        if self.n_clusters <= 0:
            LOGGER.error("n_clusters must be > 0, got %d", self.n_clusters)
            raise ValueError(f"n_clusters must be > 0, got {self.n_clusters}")
        if self.pairs_per_cluster_pair <= 0:
            LOGGER.error("Pairs per cluster pair must be > 0, got %d", self.pairs_per_cluster_pair)
            raise ValueError(f"Pairs per cluster pair must be > 0, got {self.pairs_per_cluster_pair}")
        if self.k_selected <= 0:
            LOGGER.error("k_selected must be > 0, got %d", self.k_selected)
            raise ValueError(f"k_selected must be > 0, got {self.k_selected}")
        if self.bins <= 0:
            LOGGER.error("bins must be > 0, got %d", self.bins)
            raise ValueError(f"bins must be > 0, got {self.bins}")

    def run(self) -> None:
        LOGGER.info(
            "PairsSampler start: clusters=%d, pairs_per_cluster_pair=%d, k_selected=%d, bins=%d, seed=%d",
            self.n_clusters, self.pairs_per_cluster_pair, self.k_selected, self.bins, self.seed
        )

        self._log_gpu_info()

        core_df = self._load_core_df()
        embeddings, emb_sentences = self._load_embeddings_previous()

        # Sentence list used for indexing must match embeddings
        if emb_sentences is None:
            LOGGER.error("Embeddings file %s missing sentences array '%s'.", self._files_manager.embeddings_previous,
                         self.sentence_col)
            raise ValueError(
                f"Embeddings file {self._files_manager.embeddings_previous} missing sentences array '{self.sentence_col}'.")

        # Build sentence->id mapping from core dataset (cleaned)
        sent2id = self._build_sentence_to_id(core_df)

        # Normalize embeddings for cosine (IP on normalized)
        X = self._l2_normalize(embeddings)

        # Cluster
        labels, cluster_stats_df, cluster_sizes_df = self._faiss_kmeans_cluster(X, self.n_clusters)

        # Save cluster logs
        self._save_xlsx_must_not_exist(cluster_sizes_df, self._files_manager.clusters_sizes_current)
        self._save_xlsx_must_not_exist(cluster_stats_df, self._files_manager.clusters_stats_current)

        # Build all pairs by cluster-pair sampling
        all_pairs_df = self._sample_pairs_across_cluster_pairs(
            X=X,
            sentences=emb_sentences,
            labels=labels,
            sent2id=sent2id,
        )

        # Save all pairs (pre-binning too)
        self._save_xlsx_must_not_exist(all_pairs_df, self._files_manager.all_pairs_current)

        # Assign bins based on min/max
        binned_df, bin_stats_df = assign_minmax_bins(all_pairs_df, bins=self.bins, column="cosine_norm")

        # Select k distributed across bins
        selected_df = select_bins_distributed(
            binned_df,
            k=self.k_selected,
            bins=self.bins,
            column="cosine_norm",
            random_state=self.seed,
        )

        # Keep required columns (with bin)
        selected_df = selected_df.copy()

        selected_counts = (
            selected_df["bin"]
            .value_counts(dropna=False)
            .rename("count_selected")
            .reset_index()
            .rename(columns={"index": "bin"})
        )

        bin_stats_df = (
            bin_stats_df.merge(selected_counts, on="bin", how="left")
            .assign(count_selected=lambda d: d["count_selected"].fillna(0).astype(int))
        )

        # Save bins stats
        self._save_xlsx_must_not_exist(bin_stats_df, self._files_manager.bins_stats_current)

        # Save selected pairs to the official selected_pairs_current
        self._save_xlsx_must_not_exist(selected_df, self._files_manager.selected_pairs_current)
        self.seed += 1

        LOGGER.info("PairsSampler done. saved:\n- %s\n- %s\n- %s\n- %s\n- %s",
                    self._files_manager.clusters_sizes_current,
                    self._files_manager.clusters_stats_current,
                    self._files_manager.all_pairs_current,
                    self._files_manager.bins_stats_current,
                    self._files_manager.selected_pairs_current)

    # -------------------------
    # GPU logging
    # -------------------------
    @staticmethod
    def _log_gpu_info() -> None:
        LOGGER.info("PyTorch version: %s", torch.__version__)
        LOGGER.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
        if torch.cuda.is_available():
            LOGGER.info("CUDA available. torch.version.cuda=%s", torch.version.cuda)
            n = torch.cuda.device_count()
            LOGGER.info("Visible CUDA GPUs: %d", n)
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                LOGGER.info("GPU %d: %s | CC %d.%d | VRAM %.2f GB",
                            i, props.name, props.major, props.minor, props.total_memory / (1024 ** 3))
            cur = torch.cuda.current_device()
            LOGGER.info("Current CUDA device: %d (%s)", cur, torch.cuda.get_device_name(cur))
            try:
                free_b, total_b = torch.cuda.mem_get_info(cur)
                LOGGER.info("CUDA mem: free %.2f GB / total %.2f GB",
                            free_b / (1024 ** 3), total_b / (1024 ** 3))
            except Exception:
                LOGGER.debug("torch.cuda.mem_get_info not available", exc_info=True)
        else:
            LOGGER.info("CUDA not available in torch.")

        # FAISS GPU detection
        try:
            import faiss  # type: ignore
            LOGGER.info("FAISS version: %s", getattr(faiss, "__version__", "<unknown>"))
            try:
                LOGGER.info("FAISS detected GPUs: %s", faiss.get_num_gpus())
            except Exception:
                LOGGER.info("FAISS get_num_gpus failed (maybe CPU-only build).")
        except Exception:
            LOGGER.info("FAISS not importable at GPU log stage (will error later if required).")

    # -------------------------
    # Data loading
    # -------------------------
    def _load_core_df(self) -> pd.DataFrame:
        p = Path(self._files_manager.core_dataset)
        if not p.exists():
            raise FileNotFoundError(f"core_dataset not found: {p}")
        df = load_dataframe(p)
        df = ensure_sentence_id_column(df, id_col="ID")
        df = clean_sentences_df(df, self.sentence_col, drop_empty=True, drop_duplicates=False)
        return df

    def _build_sentence_to_id(self, core_df: pd.DataFrame) -> Dict[str, Any]:
        if "ID" not in core_df.columns:
            LOGGER.error("core_df must contain 'ID' after ensure_sentence_id_column")
            raise KeyError("core_df must contain 'ID' after ensure_sentence_id_column")
        if self.sentence_col not in core_df.columns:
            LOGGER.error("core_df missing sentence_col=%s", self.sentence_col)
            raise KeyError(f"core_df missing sentence_col={self.sentence_col}")

        # If duplicates exist, keep the first ID for that sentence
        sent2id = {}
        for _, row in core_df.iterrows():
            s = str(row[self.sentence_col])
            if s not in sent2id:
                sent2id[s] = row["ID"]
        LOGGER.info("Built sentence->ID mapping size=%d", len(sent2id))
        return sent2id

    def _load_embeddings_previous(self) -> Tuple[np.ndarray, Optional[List[str]]]:
        emb_path = self._files_manager.embeddings_previous
        if emb_path is None:
            LOGGER.error("embeddings_previous not available (no previous iteration).")
            raise FileNotFoundError("embeddings_previous is None (no previous iteration).")

        emb_path = Path(emb_path)
        if not emb_path.exists():
            LOGGER.error("embeddings_previous file not found: %s", emb_path)
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if emb_path.suffix != ".npy":
            LOGGER.error("Embeddings file must be .npy. Got %s", emb_path.suffix)
            raise ValueError(f"Embeddings file must be .npy. Got {emb_path.suffix}")

        LOGGER.info("Loading embeddings from %s", emb_path)
        X = np.load(emb_path).astype(np.float32, copy=False)
        if X.ndim != 2:
            LOGGER.error("Embeddings must be 2D (N,D). Got shape=%s", X.shape)
            raise ValueError(f"Embeddings must be 2D (N,D), got shape={X.shape}")

        sent_path = self._files_manager.sentences_previous
        if sent_path is None or not sent_path.exists():
            LOGGER.error("Sentences file not found: %s", sent_path)
            raise FileNotFoundError(f"Sentences file not found: {sent_path}")

        LOGGER.info("Loading sentences from %s", sent_path)
        sentences = np.load(sent_path, allow_pickle=True).tolist()

        if len(sentences) != X.shape[0]:
            LOGGER.error(
                "Embeddings alignment error: X has %d rows but sentences has %d.",
                X.shape[0], len(sentences)
            )
            raise ValueError(
                f"Embeddings alignment error: X has {X.shape[0]} rows but sentences has {len(sentences)}. "
                "Fix by re-exporting embeddings (clean only at export, never at load)."
            )
        return X, sentences

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (x / norms).astype(np.float32, copy=False)

    # -------------------------
    # Clustering
    # -------------------------
    def _faiss_kmeans_cluster(
            self,
            X: np.ndarray,
            n_clusters: int,
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError("faiss is not available. Install faiss-gpu for GPU clustering.") from e

        if X.shape[0] < n_clusters:
            raise ValueError(f"Not enough samples N={X.shape[0]} for n_clusters={n_clusters}")

        d = X.shape[1]
        LOGGER.info("Clustering with FAISS KMeans: N=%d, dim=%d, k=%d", X.shape[0], d, n_clusters)

        # KMeans (GPU if available)
        use_gpu = False
        try:
            use_gpu = faiss.get_num_gpus() > 0
        except Exception:
            use_gpu = False

        kmeans = faiss.Kmeans(d,
                              n_clusters,
                              niter=50,
                              nredo=10,
                              spherical=True,
                              verbose=False,
                              gpu=use_gpu,
                              seed=self.seed, )
        kmeans.train(X)

        # Assign
        D, I = kmeans.index.search(X, 1)  # distances, cluster ids
        labels = I.reshape(-1).astype(int)

        # cluster sizes
        sizes = np.bincount(labels, minlength=n_clusters)
        cluster_sizes_df = pd.DataFrame({
            "cluster": np.arange(n_clusters, dtype=int),
            "count": sizes.astype(int),
        }).sort_values("cluster").reset_index(drop=True)

        # cluster stats (basic + objective)
        # D are inner products to centroid for normalized vectors (cos-like)
        cluster_stats_df = pd.DataFrame([{
            "n_clusters": int(n_clusters),
            "objective_final": float(kmeans.obj[-1]) if hasattr(kmeans, "obj") and len(kmeans.obj) else None,
            "gpu": bool(use_gpu),
            "N": int(X.shape[0]),
            "dim": int(d),
        }])

        LOGGER.info("Clustering done. sizes: min=%d, max=%d, empty=%d",
                    int(sizes.min()), int(sizes.max()), int(np.sum(sizes == 0)))
        return labels, cluster_stats_df, cluster_sizes_df

    # -------------------------
    # Pair sampling across cluster pairs
    # -------------------------
    def _sample_pairs_across_cluster_pairs(
            self,
            *,
            X: np.ndarray,
            sentences: List[str],
            labels: np.ndarray,
            sent2id: Dict[str, Any],
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)

        # indices per cluster
        cluster2idx = {c: np.where(labels == c)[0] for c in range(self.n_clusters)}

        rows: List[Dict[str, Any]] = []
        pair_id_counter = 0

        seen = set()

        for i in range(self.n_clusters):
            idx_i = cluster2idx[i]
            if len(idx_i) == 0:
                continue

            for j in range(i, self.n_clusters):  # i<=j (includes self pairs)
                idx_j = cluster2idx[j]
                if len(idx_j) == 0:
                    continue

                n_take = self.pairs_per_cluster_pair

                for _ in range(n_take):
                    a = int(rng.choice(idx_i))
                    b = int(rng.choice(idx_j))
                    if a == b:
                        continue  # skip self-pair

                    key = (a, b) if a < b else (b, a)
                    if key in seen:
                        continue
                    seen.add(key)

                    sA = sentences[a]
                    sB = sentences[b]

                    # cosine since X is normalized
                    dist = float(paired_cosine_distances(X[a:a + 1], X[b:b + 1])[0])
                    cos = 1.0 - dist  # cosine similarity in [-1,1]
                    cos_norm = float((cos + 1.0) / 2.0)  # [0,1]

                    idA = sent2id.get(sA, a)
                    idB = sent2id.get(sB, b)

                    rows.append({
                        "ID": f"pair_{self._files_manager.current_iteration:02d}_{pair_id_counter:09d}",
                        "cluster_a": int(i),
                        "cluster_b": int(j),
                        "sent_id_a": idA,
                        "sent_id_b": idB,
                        "SentenceA": sA,
                        "SentenceB": sB,
                        "cosine": cos,
                        "cosine_norm": cos_norm,
                    })
                    pair_id_counter += 1

        df = pd.DataFrame(rows)
        if len(df) == 0:
            raise RuntimeError("No pairs were sampled (unexpected).")
        LOGGER.info("Sampled total pairs=%d across cluster pairs.", len(df))
        return df

    # -------------------------
    # IO safety
    # -------------------------
    @staticmethod
    def _save_xlsx_must_not_exist(df: pd.DataFrame, path: Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        save_dataframe_single(df, path, exists_ok=False)

    # -------------------------
    # CLI
    # -------------------------
    @staticmethod
    def _parse_cli(argv: Optional[List[str]]) -> tuple[Dict[str, Any], Optional[str]]:
        if argv is None:
            return {}, None
        p = ArgumentParser(description="Cluster-based Pair Sampler", allow_abbrev=False)

        p.add_argument("--config", type=str, help="Path to config YAML")
        p.add_argument("--n-clusters", dest="n_clusters", type=int, help="Number of clusters")
        p.add_argument("--pairs-per-cluster-pair", dest="pairs_per_cluster_pair", type=int,
                       help="How many pairs to sample per cluster pair")
        p.add_argument("--k-selected", dest="k_selected", type=int,
                       help="How many total pairs to keep after bin selection")
        p.add_argument("--bins", type=int, help="Number of equal-width bins on cosine_norm")
        p.add_argument("--sentence-col", dest="sentence_col", type=str, help="Sentence column name")
        p.add_argument("--embeddings-col", dest="embeddings_col", type=str, help="Embeddings column name")
        p.add_argument("--seed", type=int, help="Random seed")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)
        maybe_config_path = raw.pop("config", None)
        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
