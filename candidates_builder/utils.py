import numpy as np
import pandas as pd
import Levenshtein

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32, copy=False)


def calculate_wylie_distance(sent1, sent2):
    # 1. Character-Level Distance
    # Counts individual keystroke differences (insertions, deletions, substitutions)
    # char_dist = Levenshtein.distance(sent1, sent2)

    # 2. Syllable-Level Distance
    # Wylie usually separates syllables with spaces or slashes.
    # We split by space to treat each syllable as a single "token".
    # Clean up standard Wylie punctuation if necessary (like removing /)
    s1_tokens = sent1.replace('/', ' ').split()
    s2_tokens = sent2.replace('/', ' ').split()

    # Levenshtein.distance can accept lists of strings, not just raw strings
    syllable_dist = Levenshtein.distance(s1_tokens, s2_tokens)

    # return char_dist, syllable_dist
    return syllable_dist


def cosine_01_from_l2(x_l2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.sum(x_l2[a] * x_l2[b], axis=1)
    # cos in [-1,1] -> normalize to [0,1]
    return ((cos + 1.0) / 2.0).astype(np.float32)


def cosine_distribution_stats(x_l2: np.ndarray, *, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = x_l2.shape[0]
    i = rng.integers(0, n, size=samples, endpoint=False)
    j = rng.integers(0, n, size=samples, endpoint=False)
    cos = np.sum(x_l2[i] * x_l2[j], axis=1)

    qs = np.quantile(cos, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    df = pd.DataFrame([{
        "pairs": int(samples),
        "mean": float(np.mean(cos)),
        "std": float(np.std(cos)),
        "min": float(np.min(cos)),
        "max": float(np.max(cos)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
    }])
    return df