from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinStats:
    bin_id: int
    count: int
    min_value: float
    max_value: float


def assign_minmax_bins(
    df: pd.DataFrame,
    *,
    bins: int,
    column: str = "cosine_norm",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign equal-width bins based on [min, max] of the column (NOT quantiles).

    Returns:
      df_with_bin, bin_stats_df
    """
    if bins <= 0:
        raise ValueError(f"bins must be > 0, got {bins}")
    if column not in df.columns:
        raise KeyError(f"Missing column='{column}' in df. Available={list(df.columns)}")

    out = df.copy()
    vals = out[column].astype(float)

    vmin = float(np.nanmin(vals.to_numpy()))
    vmax = float(np.nanmax(vals.to_numpy()))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"Non-finite min/max in column {column}. min={vmin}, max={vmax}")
    if vmax == vmin:
        LOGGER.warning("All values identical (min=max=%.6f). Putting everything in bin 0.", vmin)
        out["bin"] = 0
        stats = pd.DataFrame([{
            "bin": 0, "count": int(len(out)), "min": vmin, "max": vmax
        }])
        return out, stats

    # edges length bins+1
    edges = np.linspace(vmin, vmax, num=bins + 1, dtype=float)

    # digitize into [0..bins-1], last edge inclusive
    # np.digitize returns 1..bins, so subtract 1
    b = np.digitize(vals.to_numpy(), edges[1:-1], right=False)
    out["bin"] = b.astype(int)

    stats_rows = []
    for bi in range(int(out["bin"].min()), int(out["bin"].max()) + 1):
        g = out[out["bin"] == bi][column].astype(float)
        if len(g) == 0:
            continue
        stats_rows.append({
            "bin": int(bi),
            "count": int(len(g)),
            "min": float(g.min()),
            "max": float(g.max()),
        })

    stats = pd.DataFrame(stats_rows).sort_values("bin").reset_index(drop=True)
    return out, stats


def select_bins_distributed(
    df: pd.DataFrame,
    *,
    k: int,
    bins: int,
    column: str = "cosine_norm",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select exactly K rows.
    Step 1: Try to sample evenly across min/max-based bins.
    Step 2: If some bins are too small, fill the remainder from the global leftover pool
            (which effectively "fills other bins more") until reaching K.
    """

    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if bins <= 0:
        raise ValueError(f"bins must be > 0, got {bins}")
    if column not in df.columns:
        raise ValueError(f"column '{column}' not in df.columns: {list(df.columns)}")

    n = df.shape[0]
    if n <= k:
        LOGGER.info("Requested k=%d but df has only %d rows. Returning full df.", k, n)
        return df.copy()

    LOGGER.info(
        "Selecting k=%d rows out of n=%d using %d min/max bins on column='%s'.",
        k, n, bins, column
    )

    binned, _stats = assign_minmax_bins(df, bins=bins, column=column)
    unique_bins = sorted(pd.unique(binned["bin"]).tolist())
    if len(unique_bins) == 0:
        raise ValueError("No bins created (unexpected).")

    rng = np.random.default_rng(random_state)

    target_per_bin = k // len(unique_bins)
    parts = []
    remaining = binned

    # 1) First pass: sample ~evenly from each bin (as much as available)
    for bi in unique_bins:
        g = remaining[remaining["bin"] == bi]
        if len(g) == 0:
            continue

        take = min(target_per_bin, len(g))
        if take <= 0:
            continue

        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        parts.append(remaining.loc[idx])
        remaining = remaining.drop(index=idx)

    out = pd.concat(parts, ignore_index=True) if parts else binned.iloc[0:0].copy()
    picked = len(out)

    # 2) Fill exactly to K from whatever is left (this is the key fix)
    need = k - picked
    if need > 0:
        if len(remaining) < need:
            # With n > k, this should basically never happen unless indices got weird,
            # but keep it safe.
            LOGGER.warning(
                "Need %d more rows to reach k=%d, but only %d remain. Returning %d rows.",
                need, k, len(remaining), picked + len(remaining)
            )
            extra = remaining
        else:
            idx = rng.choice(remaining.index.to_numpy(), size=need, replace=False)
            extra = remaining.loc[idx]

        out = pd.concat([out, extra], ignore_index=True)

    # Final safety: ensure exact K
    if len(out) != k:
        # Should not happen now, but enforce deterministically.
        LOGGER.warning("Post-selection size=%d != k=%d. Forcing exact size.", len(out), k)
        out = out.sample(n=k, random_state=random_state).reset_index(drop=True)

    LOGGER.info(
        "Selected subset complete: target k=%d, actual=%d, unique_bins=%d.",
        k, len(out), len(unique_bins)
    )
    return out