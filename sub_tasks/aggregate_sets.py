
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import os
import warnings
from IPython.display import display
import argparse
from sub_tasks.concat_aggregated_results import concat_aggregated_results

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_config import AppConfig, CONFIG_FILE_NAME


def aggregate_sets(given_sets_filepath, given_aggregation_filepath):

    df_llms = pd.read_csv(given_sets_filepath)

    # Find all train filename and size columns (train_filename, train1_filename, ...; train_size, train1_size, ...)
    train_filename_cols = sorted([c for c in df_llms.columns if c.startswith('train') and c.endswith('_filename')])
    train_size_cols = sorted([c for c in df_llms.columns if c.startswith('train') and c.endswith('_size')])

    # Normalize set-specific filenames (replace _A_, _B_, _C_, _D_ with _X_) so groups match across sets
    for col in train_filename_cols + ['validation_filename', 'test_filename']:
        if col in df_llms.columns:
            df_llms[col] = df_llms[col].astype(str).replace('nan', '')
            for s in ['A', 'B', 'C', 'D']:
                df_llms[col] = df_llms[col].str.replace(f"_{s}_", "_X_", regex=False)

    # Zero out train_size values for no_fit rows before aggregation
    if 'no_fit' in df_llms.columns and train_size_cols:
        no_fit_mask = df_llms['no_fit'].astype(str).str.strip().str.lower() == 'true'
        df_llms.loc[no_fit_mask, train_size_cols] = 0

    # Group by model and training configuration
    _base_group_cols = ['model_name_or_path', 'no_fit', 'pooling_strategy', 'loss_type', "loss_scale", 'learning_rate', 'gradient_accumulation_steps', 'lr_scheduler_type', 'weight_decay', 'per_device_train_batch_size', "use_unicode_columns", "warmup_steps", "save_strategy"]
    _time_cols = [c for c in ['epoch', 'step'] if c in df_llms.columns]
    llms_group_cols = [c for c in _base_group_cols if c in df_llms.columns] + _time_cols + train_filename_cols + train_size_cols

    llms_metric_cols = [
        'train_spearman', 'train_pearson', 'train_kendall',
        'validation_spearman', 'validation_pearson', 'validation_kendall',
        'test_spearman', 'test_pearson', 'test_kendall',
        'train_score_mean', 'train_score_std',
    ]

    # Ensure metrics are numeric
    for col in llms_metric_cols:
        if col in df_llms.columns:
            df_llms[col] = pd.to_numeric(df_llms[col], errors='coerce')

    # Aggregate: mean and std across the 4 sets
    llms_agg_dict = {col: ['mean', 'std', 'count'] for col in llms_metric_cols if col in df_llms.columns}
    agg_df = df_llms.groupby(llms_group_cols, dropna=False)[list(llms_agg_dict.keys())].agg(llms_agg_dict)

    # Verify each group has exactly 4 values (one per set)
    counts = agg_df.xs('count', level=1, axis=1)
    if (counts != 4).any().any():
        print("WARNING: Some groups do not have exactly 4 values!")
        display(counts[counts != 4].dropna(how='all'))
    else:
        print("All groups have exactly 4 values (A, B, C, D).")

    # Flatten column names
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    # Capture row count from the first available count column, then drop the rest
    count_cols = [c for c in agg_df.columns if c.endswith('_count')]
    if count_cols:
        agg_df['n_rows'] = agg_df[count_cols[0]]
    agg_df = agg_df[[c for c in agg_df.columns if not c.endswith('_count')]]

    # Shorten model_name_or_path for readability
    agg_df['model_short'] = agg_df['model_name_or_path'].str.split('/').str[-1]

    # Sort by best test_spearman_mean descending
    agg_df = agg_df.sort_values('test_spearman_mean', ascending=False)

    # Save to Excel
    agg_df.to_csv(given_aggregation_filepath, index=False)
    abs_path = Path(given_aggregation_filepath).resolve()
    print(f"Saved to: file://{abs_path}")
    print(f"Result: {len(agg_df)} rows\n")

    # Display summary
    display_cols = ['model_short'] + train_filename_cols + train_size_cols + _time_cols + [
        'test_spearman_mean', 'test_spearman_std',
        'validation_spearman_mean', 'validation_spearman_std',
    ]
    display(agg_df[[c for c in display_cols if c in agg_df.columns]])


def post_active_sampling():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_time_identifier', default='2026-02-20_21-10-23', help='Run time identifier to locate the correct results files')
    parser.add_argument('--cross_trainer', action='store_true', default=False)
    args = parser.parse_args()
    run_time_identifier = args.run_time_identifier

    app_config = AppConfig(str(Path(__file__).resolve().parent.parent / CONFIG_FILE_NAME))
    sensim_dir = app_config.get('sensim_base_dir')
    cross_file_prefix = ''
    cross_sub_dir = ''
    if args.cross_trainer:
        cross_sub_dir = f"Cross-Trainer/"
        cross_file_prefix = 'cross_'

    # BASE_FOLDER = f"../results/active_sampling/{run_time_identifier}"
    # sets_filepath = f"../results/llms_sets_results_{run_time_identifier}.csv"
    # aggregation_filepath = f"../results/active_sampling/{run_time_identifier}/llms_aggregated_sets_results_{run_time_identifier}.csv"
    # shutil.copy(sets_filepath, BASE_FOLDER)

    sets_filepath =        f"{sensim_dir}/results/{cross_sub_dir}llms_{cross_file_prefix}sets_results_{run_time_identifier}.csv"
    aggregation_filepath = f"{sensim_dir}/results/{cross_sub_dir}llms_{cross_file_prefix}aggregated_sets_results_{run_time_identifier}.csv"
    aggregate_sets(sets_filepath, aggregation_filepath)

    # If epoch-level results exist, aggregate those as well
    sets_filepath_epoch = f"{sensim_dir}/results/{cross_sub_dir}llms_{cross_file_prefix}sets_results_{run_time_identifier}_epochs.csv"
    if os.path.exists(sets_filepath_epoch):
        aggregation_filepath_epoch = f"{sensim_dir}/results/{cross_sub_dir}llms_{cross_file_prefix}aggregated_sets_results_{run_time_identifier}_epochs.csv"
        aggregate_sets(sets_filepath_epoch, aggregation_filepath_epoch)

    # If step-level results exist, aggregate those as well
    sets_filepath_steps = f"{sensim_dir}/results/{cross_sub_dir}llms_{cross_file_prefix}sets_results_{run_time_identifier}_steps.csv"
    if os.path.exists(sets_filepath_steps):
        aggregation_filepath_steps = f"{sensim_dir}/results/{cross_sub_dir}llms_{cross_file_prefix}aggregated_sets_results_{run_time_identifier}_steps.csv"
        aggregate_sets(sets_filepath_steps, aggregation_filepath_steps)

    results_dir = f"{sensim_dir}/results/{cross_sub_dir}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #concat_output = f"{results_dir}/llms_aggregated_all_sets_results_{timestamp}.csv"
    concat_output = f"{results_dir}llms_{cross_file_prefix}aggregated_all_sets_results_{run_time_identifier}.csv"
    concat_aggregated_results(results_dir, concat_output)

# run using
#   python -m sub_tasks.aggregate_sets --run_time_identifier 2027-04-28_21-00-00 --cross_trainer
if __name__ == "__main__":

    post_active_sampling()
