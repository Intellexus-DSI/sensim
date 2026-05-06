"""
Visualize the distribution of sentence-transformer embeddings.

Usage:
    python -m sub_tasks.visualize_embeddings --embeddings path/to/embeddings_00.npy
    python -m sub_tasks.visualize_embeddings --embeddings path/to/embeddings_00.npy --method umap --sample 5000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(path: str):
    p = Path(path)
    embeddings = np.load(p).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D array in {path}, got shape {embeddings.shape}")

    # Try loading sibling sentences file (e.g. sentences_00.npy next to embeddings_00.npy)
    sentences = None
    sent_path = p.parent / p.name.replace("embeddings_", "sentences_")
    if sent_path.exists() and sent_path != p:
        sentences = np.load(sent_path, allow_pickle=True)

    print(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} from '{p.name}'")
    return embeddings, sentences


def reduce_2d(embeddings: np.ndarray, method: str = "pca", perplexity: int = 30):
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings), reducer.explained_variance_ratio_
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        return reducer.fit_transform(embeddings), None
    elif method == "umap":
        try:
            import umap
        except ImportError:
            sys.exit("umap-learn is not installed. Install it with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings), None
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def count_close_neighbors(embeddings: np.ndarray, threshold: float, batch_size: int = 512):
    """For each vector, count how many other vectors have cosine similarity >= threshold.

    Processes in batches to avoid allocating an (N x N) matrix at once.
    """
    n = len(embeddings)
    # Normalize once so dot product == cosine similarity
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    counts = np.zeros(n, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (batch, dim) @ (dim, N) -> (batch, N)
        sims = normed[start:end] @ normed.T
        # Count neighbours (exclude self by subtracting 1 per row)
        close = (sims >= threshold).sum(axis=1) - 1
        counts[start:end] = close

    return counts


def plot_all(embeddings: np.ndarray, method: str, output_dir: Path, perplexity: int = 30,
             cosine_sample: int = 2000, neighbor_thresholds: list = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(embeddings, axis=1)

    # --- 1. 2-D scatter ---
    coords, var_ratio = reduce_2d(embeddings, method=method, perplexity=perplexity)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.4, c=norms, cmap="viridis")
    plt.colorbar(scatter, ax=ax, label="L2 norm")
    title = f"2-D {method.upper()} projection  (n={len(embeddings)}, dim={embeddings.shape[1]})"
    if var_ratio is not None:
        title += f"\nPC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}"
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_{method}.png", dpi=150)
    plt.close(fig)
    print(f"Saved scatter_{method}.png")

    # --- 2. Embedding norm distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norms, bins=80, edgecolor="black", linewidth=0.3)
    ax.set_title(f"Embedding L2 norm distribution  (mean={norms.mean():.3f}, std={norms.std():.3f})")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "norm_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved norm_distribution.png")

    # --- 3. Pairwise cosine similarity (sampled) ---
    n = min(cosine_sample, len(embeddings))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sample = embeddings[idx]
    cos_sim = cosine_similarity(sample)
    # Take upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(n, k=1)
    cos_vals = cos_sim[triu_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cos_vals, bins=100, edgecolor="black", linewidth=0.3)
    ax.set_title(f"Pairwise cosine similarity  (sample={n}, mean={cos_vals.mean():.3f}, std={cos_vals.std():.3f})")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_similarity_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved cosine_similarity_distribution.png")

    # --- 4. Close-neighbor count per vector ---
    if neighbor_thresholds is None:
        neighbor_thresholds = [0.90, 0.95, 0.99]

    fig, axes = plt.subplots(1, len(neighbor_thresholds), figsize=(6 * len(neighbor_thresholds), 5), squeeze=False)
    axes = axes[0]
    for ax, thresh in zip(axes, neighbor_thresholds):
        counts = count_close_neighbors(embeddings, threshold=thresh)
        ax.hist(counts, bins=min(80, max(counts) - min(counts) + 1), edgecolor="black", linewidth=0.3)
        pct_isolated = (counts == 0).sum() / len(counts) * 100
        ax.set_title(f"cos >= {thresh}\nmedian={int(np.median(counts))}, "
                     f"mean={counts.mean():.1f}, isolated={pct_isolated:.1f}%")
        ax.set_xlabel("# close neighbors")
        ax.set_ylabel("Count")
    fig.suptitle("Close-neighbor count per vector", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "neighbor_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved neighbor_counts.png")

    # --- 5. Per-dimension mean / std ---
    dim_means = embeddings.mean(axis=0)
    dim_stds = embeddings.std(axis=0)
    dims = np.arange(embeddings.shape[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.bar(dims, dim_means, width=1.0, color="steelblue")
    ax1.set_ylabel("Mean")
    ax1.set_title("Per-dimension statistics")
    ax2.bar(dims, dim_stds, width=1.0, color="coral")
    ax2.set_ylabel("Std")
    ax2.set_xlabel("Dimension index")
    fig.tight_layout()
    fig.savefig(output_dir / "per_dimension_stats.png", dpi=150)
    plt.close(fig)
    print("Saved per_dimension_stats.png")

    print(f"\nAll plots saved to {output_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize sentence-transformer embeddings distribution.")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to an .npy embeddings file (produced by SbertModel._export_core_embeddings)")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne", "umap"],
                        help="Dimensionality reduction method for the 2-D scatter (default: pca)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N embeddings before plotting (0 = use all)")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="Perplexity for t-SNE (default: 30)")
    parser.add_argument("--cosine-sample", type=int, default=2000, dest="cosine_sample",
                        help="Number of embeddings to sample for pairwise cosine histogram (default: 2000)")
    parser.add_argument("--neighbor-thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.99],
                        dest="neighbor_thresholds",
                        help="Cosine similarity thresholds for the close-neighbor count plot (default: 0.90 0.95 0.99)")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir",
                        help="Directory to save plots (default: same directory as the embeddings file)")
    args = parser.parse_args(argv)

    embeddings, sentences = load_embeddings(args.embeddings)

    if args.sample > 0 and args.sample < len(embeddings):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embeddings), size=args.sample, replace=False)
        embeddings = embeddings[idx]
        if sentences is not None:
            sentences = sentences[idx]
        print(f"Sampled {args.sample} embeddings for visualization")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.embeddings).parent
    plot_all(embeddings, method=args.method, output_dir=output_dir,
             perplexity=args.perplexity, cosine_sample=args.cosine_sample,
             neighbor_thresholds=args.neighbor_thresholds)


if __name__ == "__main__":
    main()
