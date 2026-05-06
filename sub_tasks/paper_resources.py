"""
Generate LaTeX tables for academic paper from model evaluation results.

Usage:
    python -m sub_tasks.paper_tables --csv /path/to/eval_models_results.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column name mapping: CSV column -> display name in the paper
# Fill in the right-hand values as needed.
# ---------------------------------------------------------------------------
COLUMN_DISPLAY_NAMES: dict[str, str] = {
    "encoder_type":       "Encoder Type",
    "model_name_or_path": "Model",
    "test0_spearman":     "Spearman",
    "test0_pearson":      "Pearson",
}

MULTI_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "models_cell":   "Models",
    "weights_cell":  "Weights",
    "test_spearman": "Test Spearman",
    "test_pearson":  "Test Pearson",
}


def _parse_combined(encoder_type: str) -> list[tuple[str, str]]:
    """Parse 'combined(name1=w1,name2=w2,...)' into [(name1, w1), ...]."""
    m = re.match(r"combined\((.+)\)$", str(encoder_type))
    if not m:
        return [(encoder_type, "")]
    pairs = []
    for item in m.group(1).split(","):
        if "=" in item:
            name, weight = item.rsplit("=", 1)
            pairs.append((name.strip(), weight.strip()))
        else:
            pairs.append((item.strip(), ""))
    return pairs


def _find_metric_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column that exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of {candidates} found in CSV columns: {list(df.columns)}")


def generate_single_model_latex_table(csv_path: str | Path) -> str:
    """Load the eval CSV, keep only single-model rows, and return a LaTeX table string.

    Args:
        csv_path: Path to the evaluation results CSV.

    Returns:
        A LaTeX ``booktabs``-style table as a string.
    """
    df = pd.read_csv(csv_path)

    # Keep only single-model rows (exclude combined/ensemble entries)
    is_combined = df["encoder_type"].astype(str).str.startswith("combined")
    df = df[~is_combined].copy()

    # Resolve the actual spearman / pearson column names (handles both naming schemes)
    spearman_col = _find_metric_col(df, ["test0_spearman", "test_spearman"])
    pearson_col  = _find_metric_col(df, ["test0_pearson",  "test_pearson"])

    # Select and rename columns
    selected = df[["encoder_type", "model_name_or_path", spearman_col, pearson_col]].copy()
    selected = selected.rename(columns={
        spearman_col: "test0_spearman",
        pearson_col:  "test0_pearson",
    })

    display_cols = list(COLUMN_DISPLAY_NAMES.keys())
    selected = selected[display_cols]

    # Round metrics to 4 decimal places
    selected["test0_spearman"] = selected["test0_spearman"].round(4)
    selected["test0_pearson"]  = selected["test0_pearson"].round(4)

    return _to_latex(selected)


def generate_multi_model_latex_table(csv_path: str | Path) -> str:
    """Load the eval CSV, keep only combined/ensemble rows, and return a LaTeX table string.

    Args:
        csv_path: Path to the evaluation results CSV.

    Returns:
        A LaTeX ``booktabs``-style table as a string.
    """
    df = pd.read_csv(csv_path)

    is_combined = df["encoder_type"].astype(str).str.startswith("combined")
    df = df[is_combined].copy()

    for col in ["test_spearman", "test_pearson"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    def _make_cells(encoder_type: str) -> tuple[str, str]:
        pairs = _parse_combined(encoder_type)
        names   = r" \\ ".join(p[0].replace("_", r"\_") for p in pairs)
        weights = r" \\ ".join(p[1] for p in pairs)
        return f"\\makecell[l]{{{names}}}", f"\\makecell[r]{{{weights}}}"

    # Drop rows where any component has weight 0
    df = df[~df["encoder_type"].apply(
        lambda et: any(float(w) == 0.0 for _, w in _parse_combined(et) if w)
    )].copy()

    cells = pd.DataFrame(
        df["encoder_type"].apply(_make_cells).tolist(),
        index=df.index,
        columns=["models_cell", "weights_cell"],
    )
    selected = pd.concat([cells, df[["test_spearman", "test_pearson"]]], axis=1)
    selected = selected.sort_values("test_spearman", ascending=False)

    return _to_latex_multi(selected)


def _to_latex_multi(df: pd.DataFrame) -> str:
    """Convert a multi-model DataFrame to a two-column-spanning LaTeX booktabs table."""
    metric_cols = {"test_spearman", "test_pearson"}
    raw_cols    = {c for c in df.columns if c.endswith("_cell")}
    display_names = [
        r"\textbf{" + MULTI_MODEL_DISPLAY_NAMES.get(c, c) + r"}"
        for c in df.columns
    ]
    # models_cell → left-aligned; weights + metrics → right-aligned
    col_fmt = "l" + "r" * (len(df.columns) - 1)

    lines = [
        r"% Requires \usepackage{makecell} in preamble",
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Multi-Model Ensemble Evaluation Results}",
        r"  \label{tab:multi_model_results}",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(display_names) + r" \\",
        r"    \midrule",
    ]

    for _, row in df.iterrows():
        cells = []
        for col, val in zip(df.columns, row):
            if col in metric_cols:
                cells.append(f"{val:.4f}" if pd.notna(val) else "--")
            elif col in raw_cols:
                cells.append(str(val))  # pre-formatted LaTeX, no escaping
            else:
                cells.append(str(val).replace("_", r"\_").replace("%", r"\%"))
        lines.append("    " + " & ".join(cells) + r" \\ \hline")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


def _to_latex(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a LaTeX booktabs table."""
    display_names = [COLUMN_DISPLAY_NAMES.get(c, c) for c in df.columns]
    col_fmt = "l" * len(df.columns)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Single-Model Evaluation Results}",
        r"  \label{tab:single_model_results}",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(display_names) + r" \\",
        r"    \midrule",
    ]

    for _, row in df.iterrows():
        cells = []
        for col, val in zip(df.columns, row):
            if col in ("test0_spearman", "test0_pearson"):
                cells.append(f"{val:.4f}")
            else:
                # Escape underscores and % signs for LaTeX
                cells.append(str(val).replace("_", r"\_").replace("%", r"\%"))
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from eval results CSV.")
    parser.add_argument(
        "--csv",
        default="/home/shailu1492/repositories/intellexus-model/sensim/results/local_models_eval_results_2026-02-16_08-02-28.csv",
        help="Path to the evaluation results CSV file.",
    )
    parser.add_argument(
        "--output",
        default="/home/shailu1492/repositories/intellexus-model/sensim/results/publish/single_model_table.tex",
        help="Path to write the single-model LaTeX table.",
    )
    parser.add_argument(
        "--output-multi",
        default="/home/shailu1492/repositories/intellexus-model/sensim/results/publish/multi_model_table.tex",
        help="Path to write the multi-model ensemble LaTeX table.",
    )
    args = parser.parse_args()

    latex_single = generate_single_model_latex_table(args.csv)
    if args.output:
        Path(args.output).write_text(latex_single, encoding="utf-8")
        print(f"Single-model LaTeX table written to: {args.output}")
    else:
        print(latex_single)

    latex_multi = generate_multi_model_latex_table(args.csv)
    if args.output_multi:
        Path(args.output_multi).write_text(latex_multi, encoding="utf-8")
        print(f"Multi-model LaTeX table written to: {args.output_multi}")
    else:
        print(latex_multi)


if __name__ == "__main__":
    main()
