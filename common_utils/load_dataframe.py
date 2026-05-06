from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


SUPPORTED_TABULAR_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv"}


def load_dataframe(
    filepath: Path,
    *,
    sheet_name: Optional[Union[str, int]] = None,
    all_sheets: bool = False,
    add_sheet_column: bool = False,
    sheet_column_name: str = "__sheet__",
) -> pd.DataFrame:
    """
    Load a pandas DataFrame from CSV/TSV/Excel.

    Rules:
      1) CSV/TSV: load normally (sheet params are ignored/validated).
      2) Excel: either load a specific sheet (sheet_name) OR load all sheets and concat.

    Args:
        filepath: Path to the tabular file.
        sheet_name: Excel sheet name or index. If None and all_sheets=False, loads first sheet.
        all_sheets: If True (Excel only), loads all sheets and concatenates them.
        add_sheet_column: When all_sheets=True, add a column indicating the source sheet.
        sheet_column_name: Name of that column.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If suffix is unsupported or args are invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"The file does not exist: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(
            f"Unsupported file format for {filepath}. Supported: {sorted(SUPPORTED_TABULAR_SUFFIXES)}"
        )

    if suffix in {".csv", ".tsv"}:
        if all_sheets:
            raise ValueError("all_sheets is only valid for Excel files.")
        if sheet_name is not None:
            raise ValueError("sheet_name is only valid for Excel files.")

        if suffix == ".csv":
            return pd.read_csv(filepath)
        return pd.read_csv(filepath, sep="\t")

    # Excel (.xlsx / .xls)
    if all_sheets and sheet_name is not None:
        raise ValueError("Use either sheet_name or all_sheets=True, not both.")

    if all_sheets:
        sheets: dict[str, pd.DataFrame] = pd.read_excel(
            filepath, engine="openpyxl", sheet_name=None
        )

        if add_sheet_column:
            parts = []
            for name, sdf in sheets.items():
                tmp = sdf.copy()
                tmp[sheet_column_name] = name
                parts.append(tmp)
            return pd.concat(parts, ignore_index=True)

        return pd.concat(sheets.values(), ignore_index=True)

    return pd.read_excel(filepath, engine="openpyxl", sheet_name=(sheet_name or 0))


def load_excel_sheets_dict(filepath: Path) -> dict[str, pd.DataFrame]:
    """
    Load an Excel file into a dict of DataFrames keyed by sheet name.

    Args:
        filepath: Path to an .xlsx/.xls file.

    Returns:
        dict[sheet_name, DataFrame]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If suffix is not .xlsx/.xls.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"The file does not exist: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError("load_excel_sheets_dict only supports Excel files (.xlsx/.xls).")

    return pd.read_excel(filepath, engine="openpyxl", sheet_name=None)