import pandas as pd
import os
import re
from pathlib import Path
import shutil

from app_config import AppConfig, CONFIG_FILE_NAME

app_config = AppConfig(f'../{CONFIG_FILE_NAME}')


def merge_excel_files(base_folder_path, output_filename="merged_llms_scored.xlsx"):
    """
    Merges excel files from subfolders named it_XX into a single file.
    Assumes file naming convention: it_01 -> llms_pairs_scored_01.xlsx
    """
    base_path = Path(base_folder_path)

    # Check if base path exists
    if not base_path.exists():
        print(f"Error: Base folder not found at {base_path}")
        return

    all_dataframes = []

    # 1. Find all subfolders starting with 'it_'
    # We sort them to ensure the merge happens in order (01, 02, 03...)
    subfolders = sorted([f for f in base_path.iterdir() if f.is_dir() and f.name.startswith("it_")])

    print(f"Found {len(subfolders)} iteration folders.")

    for folder in subfolders:
        try:
            # 2. Extract the index (e.g., '01' from 'it_01')
            match = re.search(r'it_(\d+)', folder.name)

            if not match:
                print(f"Skipping {folder.name}: Could not extract index number.")
                continue

            index_str = match.group(1)  # e.g., "01"

            # 3. Construct the expected filename
            target_file = f"llms_pairs_scored_{index_str}.xlsx"
            file_path = folder / target_file

            # 4. Read the file if it exists
            if file_path.exists():
                print(f"Processing: {file_path}")
                df = pd.read_excel(file_path)

                # Optional: Add a column to track which iteration this data came from
                df['iteration_source'] = folder.name

                all_dataframes.append(df)
            else:
                print(f"Warning: Expected file not found: {file_path}")

        except Exception as e:
            print(f"Error processing folder {folder.name}: {e}")

    # 5. Concatenate and Save
    if all_dataframes:
        print("-" * 30)
        print("Merging data...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)

        output_path = base_path / output_filename
        merged_df.to_excel(output_path, index=False)
        print(f"Done! Merged {len(merged_df)} rows into: {output_path}")
    else:
        print("No matching files were found to merge.")


def generate_consecutive_pairs_from_segments(segments_filepath, output_filepath):
    """
    Generate (anchor, positive) pairs from consecutive text segments.
    Consecutive segments within the same Source_Line_Number are paired.
    Uses the Segmented_Text_EWTS column.
    """
    df = pd.read_excel(segments_filepath, engine='openpyxl')
    df = df.sort_values(['Source_Line_Number', 'Sentence_Order']).reset_index(drop=True)

    pairs = []
    for _, group in df.groupby('Source_Line_Number'):
        texts = group['Segmented_Text_EWTS'].tolist()
        for i in range(len(texts) - 1):
            if pd.notna(texts[i]) and pd.notna(texts[i + 1]):
                pairs.append({'SentenceA': texts[i], 'SentenceB': texts[i + 1]})

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_excel(output_filepath, index=False)
    print(f"Generated {len(pairs_df)} consecutive segment pairs -> {output_filepath}")
    return output_filepath


def copy_to_data_folder(base_folder, output_filename):
    output_filepath = os.path.join(base_folder, output_filename)
    data_folder = "../data/NewDataA-D"

    shutil.copy(output_filepath, data_folder)


if __name__ == "__main__":

    # --- CONFIGURATION ---
    run_time_identifier = "2026-02-18_07-32-22"  # Update this to match your folder's timestamp

    base_dir = app_config.get("sensim_base_dir", default="..")

    active_sampling_dir = f"{base_dir}/results/active_sampling/{run_time_identifier}"
    output_filename = f"merged_llms_scored_{run_time_identifier}.xlsx"

    merge_excel_files(active_sampling_dir, output_filename)

    do_copy_to_data_folder = True  # Set to True to copy the merged file to the data folder
    if do_copy_to_data_folder:
        copy_to_data_folder(active_sampling_dir, output_filename)

    # --- Generate consecutive segment pairs from merged segments ---
    # do_generate_consecutive_pairs = True
    # if do_generate_consecutive_pairs:
    #     segments_file = "../data/merged_kangyur_tengyur_segments_v2.xlsx"
    #     consecutive_pairs_output = "../data/NewDataA-D/consecutive_segment_pairs_v2.xlsx"
    #     generate_consecutive_pairs_from_segments(segments_file, consecutive_pairs_output)
