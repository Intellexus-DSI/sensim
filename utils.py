import sys
import time
import os
import gc
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from datasets import Dataset


def reorder_dict_keys(d, keyword='filename'):
    """
    Reorder dictionary keys so that all keys containing 'keyword'
    are sorted alphabetically and placed starting from the first
    keyword key's position.
    """
    all_keys = list(d.keys())
    keyword_keys = [k for k in all_keys if keyword.lower() in str(k).lower()]

    if not keyword_keys:
        return d

    first_idx = min(all_keys.index(k) for k in keyword_keys)
    other_keys = [k for k in all_keys if keyword.lower() not in str(k).lower()]

    new_order = (
            other_keys[:first_idx] +
            sorted(keyword_keys) +
            other_keys[first_idx:]
    )

    return {k: d[k] for k in new_order}


def log_evaluation_results(log_file, settings, results):
    """
    Logs experiment settings and results to a CSV file.

    If a row with the same settings already exists, it's overwritten
    with new results and a new timestamp. Otherwise, a new row is appended.

    Args:
        log_file (str): Path to the CSV log file.
        settings (dict): Dictionary of settings that identify the run (e.g., l2_reg).
        results (dict): Dictionary of results to log (e.g., accuracy).
    """
    settings = reorder_dict_keys(settings, 'filename')

    # 1. Prepare the new row data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row_data = {**settings, **results, 'timestamp': timestamp}

    # 2. Read the existing log file or create a new DataFrame
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=list(new_row_data.keys()))

    # 3. Find if a row with these exact settings already exists
    mask = pd.Series([True] * len(df))
    df_cols = set(df.columns)
    for key, value in settings.items():
        if key not in df_cols:
            mask[:] = False
            break
        mask &= df[key].isna() if pd.isna(value) else (df[key] == value)

    matching_indices = df[mask].index

    # 4. Overwrite or Append
    if not matching_indices.empty:
        index_to_update = matching_indices[0]
        for key, value in new_row_data.items():
            df.loc[index_to_update, key] = value
        print(f"Log file updated for settings: {settings}")
    else:
        new_row_df = pd.DataFrame([new_row_data])
        df = new_row_df if df.empty else pd.concat([df, new_row_df], ignore_index=True)
        print(f"New entry added to log for settings: {settings}")

    # 5. Ensure all columns are present (handles multi-fit runs adding new columns)
    all_keys = list(new_row_data.keys())
    for col in df.columns:
        if col not in all_keys:
            all_keys.append(col)
    df = df.reindex(columns=all_keys)

    # 6. Save the updated DataFrame back to CSV
    df.to_csv(log_file, index=False)


def get_dataset(filepath, df_mapping, random_state):
    df = load_dataframe(filepath)

    # Fall back to non-unicode column name when the unicode variant is absent
    effective_mapping = {}
    for col, target in df_mapping.items():
        if col not in df.columns and col.endswith('_unicode'):
            fallback = col[: -len('_unicode')]
            if fallback in df.columns:
                print(f"[get_dataset] '{col}' not found in {filepath}, falling back to '{fallback}'")
                effective_mapping[fallback] = target
                continue
        effective_mapping[col] = target

    df_columns = list(effective_mapping.keys())
    df = df[df_columns]
    df.rename(columns=effective_mapping, inplace=True)
    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    return dataset


def load_dataframe(filepath, sheet_name=None) -> pd.DataFrame:
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.xlsx':
        if sheet_name:
            df = pd.read_excel(filepath, engine='openpyxl', sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath, engine='openpyxl', sheet_name=0)
    elif filepath.suffix.lower() == '.csv':
        with open(filepath, 'r', encoding='utf-8') as _f:
            _first_line = _f.readline()
        _sep = '\t' if '\t' in _first_line else ','
        df = pd.read_csv(filepath, sep=_sep)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")

    return df


def save_dataframe(df: pd.DataFrame, filepath) -> None:
    if filepath.lower().endswith('.xlsx'):
        df.to_excel(filepath, index=False, engine='openpyxl')
    else:
        df.to_csv(filepath, encoding='utf-8-sig', index=False, sep='\t')


def select_linspace_distributed(df, k, column='cosine'):
    """
    Select k rows from dataframe with evenly distributed values in the specified column.

    Parameters:
    - df: DataFrame to sample from
    - k: Number of rows to select
    - column: Column name to distribute evenly (default: 'cosine')
    """
    # Sort by the cosine column
    df_sorted = df.sort_values(by=column).reset_index(drop=True)

    # Calculate evenly spaced indices
    indices = np.linspace(0, len(df_sorted) - 1, k).astype(int)

    # Select rows at those indices
    return df_sorted.iloc[indices]


def select_bins_distributed(df, k=1000, bins=10, column='cosine', random_state=42):
    """
    Selects a subset of K rows from the DataFrame such that the values
    in the specified column are evenly distributed across 'bins' quantiles.

    Args:
        df (pd.DataFrame): The input DataFrame.
        k (int): The target number of rows to select (the subset size).
        bins (int): The number of equal-frequency bins (quantiles) to create.
        column (str): The name of the column to base the distribution on.
        random_state (int): Seed for reproducibility of sampling.

    Returns:
        pd.DataFrame: A subset of the DataFrame with approximately K rows.
        list(int): The indices of the selected rows in the original DataFrame.
    """
    if df.shape[0] < k:
        print(
            f"Warning: DataFrame has only {df.shape[0]} rows, which is less than the requested K={k}. Returning entire DataFrame.")
        return df.copy()

    if k < bins:
        print(f"Warning: Requested K={k} is less than 'bins'={bins}. Setting bins to K.")
        bins = k

    # 1. Create quantile bins (equal number of rows in each bin)
    # 'duplicates="drop"' handles cases where many rows have the same 'cosine' value.
    try:
        df['bin'] = pd.qcut(df[column], q=bins, labels=False, duplicates='drop')
    except ValueError as e:
        # Catch the case where there are too many duplicate values for the number of bins
        print(f"Error during qcut: {e}. Trying a smaller number of bins.")
        # Fallback: Try a smaller number of bins, e.g., using a fixed smaller value or df[column].nunique()
        bins = min(bins, df[column].nunique() // 2 or 1)  # Fallback heuristic
        if bins == 0:
            return df.sample(k, random_state=random_state)

        df['bin'] = pd.qcut(df[column], q=bins, labels=False, duplicates='drop')

    # Recalculate parameters based on the *actual* number of bins created
    unique_bins = df['bin'].nunique()
    if unique_bins == 0:
        # This should be caught by the previous check, but as a safeguard
        print("Warning: Could not create any unique bins. Taking with linspace")
        return df.sample(k, random_state=random_state)

    samples_per_bin = k // unique_bins
    remaining_samples = k % unique_bins

    # 2. Sample 'samples_per_bin' rows from each bin
    # Use observed=True to only group by categories that actually appear in the data
    subset_df = df.groupby('bin', observed=True, group_keys=False).apply(lambda g: g.sample(
        n=samples_per_bin,
        replace=False,
        random_state=random_state
    ))

    # 3. Handle the remaining samples due to non-perfect division (k % unique_bins)
    if remaining_samples > 0:
        # Identify the rows already sampled
        sampled_indices = subset_df.index
        # Get the remaining rows that haven't been sampled
        remaining_df = df.drop(index=sampled_indices).dropna(subset=['bin'])

        # Take the remaining samples using a regular random sample from the unsampled rows
        extra_samples = remaining_df.sample(n=remaining_samples, replace=False, random_state=random_state)

        # Combine the main subset with the extra samples
        subset_df = pd.concat([subset_df, extra_samples], ignore_index=True)

    # Final cleanup
    final_subset = subset_df.drop(columns='bin')

    # Verification of the result size
    final_rows = final_subset.shape[0]
    print(f"✅ Subset selection complete. Target rows: {k}, Actual rows: {final_rows}")

    return final_subset


def split_input_by_distribution(input_file, output_selected, output_not_selected, k, bins, column='cosine'):
    """
    Opens an Excel file, selects k rows with evenly distributed values,
    and saves selected and non-selected rows to separate files.

    Parameters:
    - input_file: Path to input Excel file
    - output_selected: Path to save selected rows
    - output_not_selected: Path to save non-selected rows
    - k: Number of rows to select
    - column: Column name to distribute evenly (default: 'cosine')
    """
    # Read the Excel file
    df = load_dataframe(input_file)

    # Select evenly distributed rows
    selected_df = select_bins_distributed(df, k, bins, column)

    # Get non-selected rows by using set difference on index
    not_selected_df = df[~df.index.isin(selected_df.index)]

    # Removing helper columns
    # selected_df.drop(columns=['cosine', 'bin'], inplace=True, errors='ignore')
    # not_selected_df.drop(columns=['cosine', 'bin'], inplace=True, errors='ignore')

    # Save both files
    save_dataframe(selected_df, output_selected)
    save_dataframe(not_selected_df, output_not_selected)

    print(f"Selected {len(selected_df)} rows saved to: {output_selected}")
    print(f"Non-selected {len(not_selected_df)} rows saved to: {output_not_selected}")

    return selected_df, not_selected_df


def concatenate_csv_xls_files(file_list, output_file):
    # Read all CSVs and concatenate
    # df_list = [load_dataframe(file) for file in file_list] # Handles tsv files tab delimited.
    df_list = [pd.read_csv(file) for file in file_list]  # Handles csv files comma delimited.
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save to new file
    save_dataframe(combined_df, output_file)
