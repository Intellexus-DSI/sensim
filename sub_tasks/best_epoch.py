"""
best_epoch.py

For each (model, training configuration) combination in a per-epoch results CSV:
  1. Normalize set-specific filenames (A/B/C/D -> X) — same as aggregate_sets.py
  2. Aggregate (mean/std) across the 4 sets per epoch — same as aggregate_sets.py
  3. Find the epoch with the best mean validation_spearman

Usage:
    python sub_tasks/best_epoch.py --results_file results/llms_sets_results_2026-02-27_22-20-37_5000_epochs.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_best_epochs(results_file: str) -> pd.DataFrame:
    df = pd.read_csv(results_file)

    # Keep only per-epoch rows (evaluated_model == 'epoch_N')
    epoch_mask = df['evaluated_model'].astype(str).str.match(r'^epoch_\d+$')
    df = df[epoch_mask].copy()

    if df.empty:
        print("No epoch rows found in the file.")
        return pd.DataFrame()

    # --- mirror aggregate_sets.py from here ---

    # Find all train filename and size columns
    train_filename_cols = sorted([c for c in df.columns if c.startswith('train') and c.endswith('_filename')])
    train_size_cols = sorted([c for c in df.columns if c.startswith('train') and c.endswith('_size')])

    # Normalize set-specific filenames (replace _A_, _B_, _C_, _D_ with _X_) so groups match across sets
    test_filename_cols = sorted([c for c in df.columns if c.startswith('test') and c.endswith('_filename')])
    for col in train_filename_cols + ['validation_filename'] + test_filename_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '')
            for s in ['A', 'B', 'C', 'D']:
                df[col] = df[col].str.replace(f"_{s}_", "_X_", regex=False)

    # Zero out train_size values for no_fit rows before aggregation
    if 'no_fit' in df.columns and train_size_cols:
        no_fit_mask = df['no_fit'].astype(str).str.strip().str.lower() == 'true'
        df.loc[no_fit_mask, train_size_cols] = 0

    # Group by model and training configuration — same base cols as aggregate_sets.py, plus epoch
    _base_group_cols = ['model_name_or_path', 'no_fit', 'pooling_strategy', 'loss_type', 'loss_scale', 'learning_rate', 'gradient_accumulation_steps', 'lr_scheduler_type', 'weight_decay', 'per_device_train_batch_size', 'use_unicode_columns']
    group_cols = [c for c in _base_group_cols if c in df.columns] + train_filename_cols + train_size_cols + ['epoch']

    llms_metric_cols = [
        'train_spearman', 'train_pearson', 'train_kendall',
        'validation_spearman', 'validation_pearson', 'validation_kendall',
        'test_spearman', 'test_pearson', 'test_kendall',
        'train_score_mean', 'train_score_std',
    ]

    # Ensure metrics are numeric
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    for col in llms_metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Aggregate: mean and std across the 4 sets, per epoch
    llms_agg_dict = {col: ['mean', 'std', 'count'] for col in llms_metric_cols if col in df.columns}
    agg_df = df.groupby(group_cols, dropna=False)[list(llms_agg_dict.keys())].agg(llms_agg_dict)

    # Verify each group has exactly 4 values (one per set)
    counts = agg_df.xs('count', level=1, axis=1)
    if (counts != 4).any().any():
        print("WARNING: Some groups do not have exactly 4 values (A, B, C, D)!")
        print(counts[counts != 4].dropna(how='all'))
    else:
        print("All groups have exactly 4 values (A, B, C, D).")

    # Flatten column names
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    # Capture row count from the first available count column, then drop the rest
    count_cols = [c for c in agg_df.columns if c.endswith('_count')]
    if count_cols:
        agg_df['n_sets'] = agg_df[count_cols[0]]
    agg_df = agg_df[[c for c in agg_df.columns if not c.endswith('_count')]]

    # --- now find the best epoch per (model, config) ---

    model_group_cols = [c for c in _base_group_cols if c in agg_df.columns] + train_filename_cols + train_size_cols
    idx_best = agg_df.groupby(model_group_cols, dropna=False)['validation_spearman_mean'].idxmax()
    best_df = agg_df.loc[idx_best].reset_index(drop=True)

    # Shorten model name for readability
    best_df['model_short'] = best_df['model_name_or_path'].str.split('/').str[-1]

    # Sort by best validation_spearman_mean descending
    best_df = best_df.sort_values('validation_spearman_mean', ascending=False)

    return best_df


def main():
    parser = argparse.ArgumentParser(description='Find the best epoch per model/training-file combination (aggregated across A/B/C/D sets).')
    parser.add_argument(
        '--results_file',
        default='results/llms_sets_results_2026-02-27_22-20-37_5000_epochs.csv',
        help='Path to the per-epoch results CSV',
    )
    parser.add_argument(
        '--output_file',
        default=None,
        help='Optional path to save the output CSV',
    )
    args = parser.parse_args()

    result = find_best_epochs(args.results_file)

    if result.empty:
        return

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    display_cols = ['model_short'] + \
                   sorted([c for c in result.columns if c.startswith('train') and c.endswith('_filename')]) + \
                   ['epoch', 'validation_spearman_mean', 'validation_spearman_std',
                    'test_spearman_mean', 'test_spearman_std']
    display_cols = [c for c in display_cols if c in result.columns]
    print(result[display_cols].to_string(index=False))

    if args.output_file:
        result.to_csv(args.output_file, index=False)
        print(f"\nSaved to: {args.output_file}")


if __name__ == '__main__':
    main()
