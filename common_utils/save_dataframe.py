from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Mapping

import pandas as pd


SUPPORTED_TABULAR_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv"}

_EXCEL_MAX_ROWS = 1_048_576
_EXCEL_MAX_COLS = 16_384


def _sanitize_excel_sheet_name(name: str) -> str:
    """
    Excel sheet name rules:
    - max 31 chars
    - cannot contain: : \\ / ? * [ ]
    - cannot be empty
    """
    name = str(name)
    name = re.sub(r"[:\\/?*\[\]]", "_", name).strip()
    if not name:
        name = "sheet"
    return name[:31]


def _excel_write_df_split(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    base_sheet_name: str,
) -> None:
    """
    Write df to Excel. If it exceeds Excel row limit, split to multiple sheets:
      base_0, base_1, ...
    """
    if df.shape[1] > _EXCEL_MAX_COLS:
        raise ValueError(f"Too many columns for Excel: {df.shape[1]} > {_EXCEL_MAX_COLS}.")

    base = _sanitize_excel_sheet_name(base_sheet_name)
    if len(df) <= _EXCEL_MAX_ROWS:
        df.to_excel(writer, sheet_name=base, index=False)
        return

    # Split across sheets (keep header per sheet)
    chunk = _EXCEL_MAX_ROWS - 1
    n = len(df)
    parts = math.ceil(n / chunk)

    for i in range(parts):
        start = i * chunk
        end = min((i + 1) * chunk, n)
        sheet = _sanitize_excel_sheet_name(f"{base}_{i}")
        df.iloc[start:end].to_excel(writer, sheet_name=sheet, index=False)


def save_dataframe_single(
    df: pd.DataFrame,
    filepath: Path,
    *,
    exists_ok: bool = False,
    excel_sheet_name: str = "data",
) -> None:
    """
    Save a single DataFrame to CSV/TSV/Excel.
    - CSV/TSV: one file
    - Excel: tries one sheet; if too many rows, splits into sheets: <excel_sheet_name>_0, <excel_sheet_name>_1, ...

    Raises:
        FileExistsError, ValueError
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(
            f"Unsupported file format for {filepath}. Supported: {sorted(SUPPORTED_TABULAR_SUFFIXES)}"
        )

    if filepath.exists() and not exists_ok:
        raise FileExistsError(f"The file already exists: {filepath}")

    os.makedirs(filepath.parent, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(filepath, index=False)
        return
    if suffix == ".tsv":
        df.to_csv(filepath, index=False, sep="\t")
        return

    # Excel (.xlsx / .xls via openpyxl)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        _excel_write_df_split(writer, df, excel_sheet_name)


def save_dataframes_dict(
    dfs: Mapping[str, pd.DataFrame],
    filepath: Path,
    *,
    exists_ok: bool = False,
    concat: bool = False,
    concat_sheet_name: str = "data",
    csv_tsv_output_dir: Path | None = None,
) -> None:
    """
    Save a dictionary of DataFrames.

    If concat=True:
        - concatenate all dfs (in dict iteration order) and delegate to save_dataframe_single.

    If concat=False:
        - CSV/TSV: write MANY files (one per key) into a directory
          * If csv_tsv_output_dir is None, uses: <filepath_without_suffix>_parts/
          * Filenames: <key>.csv or <key>.tsv
        - Excel: write each df to its own sheet named by key; if df exceeds row limit,
          it is split to sheets: <key>_0, <key>_1, ...

    Notes:
        - For Excel, sheet names are sanitized and limited to 31 chars.
        - For CSV/TSV, filenames are sanitized (very lightly) to avoid path issues.

    Raises:
        FileExistsError, ValueError
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(
            f"Unsupported file format for {filepath}. Supported: {sorted(SUPPORTED_TABULAR_SUFFIXES)}"
        )

    if concat:
        df = pd.concat(list(dfs.values()), ignore_index=True)
        save_dataframe_single(
            df,
            filepath,
            exists_ok=exists_ok,
            excel_sheet_name=concat_sheet_name,
        )
        return

    # concat=False
    if suffix in {".csv", ".tsv"}:
        # CSV/TSV => many files
        if csv_tsv_output_dir is None:
            stem = filepath.with_suffix("").name
            csv_tsv_output_dir = filepath.parent / f"{stem}_parts"

        out_dir = Path(csv_tsv_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # If exists_ok=False and directory not empty, be strict
        if not exists_ok and any(out_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {out_dir}")

        sep = "\t" if suffix == ".tsv" else ","
        ext = suffix

        def _safe_filename(filename: str) -> str:
            filename = str(filename).strip()
            filename = re.sub(r"[^\w\-. ]+", "_", filename)  # keep letters/digits/_/-/./space
            filename = filename.replace(" ", "_")
            return filename or "data"

        for name, df in dfs.items():
            out_path = out_dir / f"{_safe_filename(name)}{ext}"
            if out_path.exists() and not exists_ok:
                raise FileExistsError(f"The file already exists: {out_path}")
            df.to_csv(out_path, index=False, sep=sep)

        return

    # Excel => one file, many sheets (split oversize sheets)
    if filepath.exists() and not exists_ok:
        raise FileExistsError(f"The file already exists: {filepath}")
    os.makedirs(filepath.parent, exist_ok=True)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for name, df in dfs.items():
            # Each df goes to sheet "name" if it fits else "name_0", "name_1", ...
            _excel_write_df_split(writer, df, str(name))