import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from common_utils import save_dataframe_single
from utils import load_dataframe


def log_evaluation_results(log_path: Path, settings: Dict[str, Any], results: Dict[str, Any]) -> None:
    """
    Logs experiment settings and results to a CSV file.

    If a row with the same settings already exists, it's overwritten
    with new results and a new timestamp. Otherwise, a new row is appended.

    Args:
        log_path (Path): Path to the CSV log file.
        settings (dict): Dictionary of settings that identify the run (e.g., l2_reg).
        results (dict): Dictionary of results to log (e.g., accuracy).
    """
    # 1. Prepare the new row data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row_data = {**settings, **results, 'timestamp': timestamp}

    # 2. Read the existing log file or create a new DataFrame
    if os.path.exists(log_path):
        df = load_dataframe(log_path)
    else:
        # Create an empty DataFrame with the correct columns if file doesn't exist
        df = pd.DataFrame(columns=list(new_row_data.keys()))

    # 3. Find if a row with these exact settings already exists
    mask = pd.Series([True] * len(df))
    for key, value in settings.items():
        if key not in df.columns:
            mask[:] = False
            break

        mask &= df[key].isna() if pd.isna(value) else (df[key] == value)

    matching_indices = df[mask].index

    # 4. Overwrite or Append
    if not matching_indices.empty:
        # A. Match found: Overwrite the first matching row
        index_to_update = matching_indices[0]
        for key, value in new_row_data.items():
            df.loc[index_to_update, key] = value
    else:
        # A. Create a DataFrame for the new row
        new_row_df = pd.DataFrame([new_row_data])

        # B. Check if the original DataFrame is empty
        if df.empty:
            # If so, the new DataFrame is just the new row
            df = new_row_df
        else:
            # Otherwise, concatenate as before
            df = pd.concat([df, new_row_df], ignore_index=True)

    # 5. Save the updated DataFrame back to CSV
    save_dataframe_single(df, log_path, exists_ok=True)
