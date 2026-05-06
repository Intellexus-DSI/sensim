import sys
import time
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import utils


from app_config import AppConfig

app_config = AppConfig()
BWS_DIR = app_config.get("scoring_script_dir", "/home/shay/Best-Worst-Scaling-Scripts")

def add_missing_columns(df, columns, default_value=None):
    """Add columns to df if they don't exist (in-place)."""
    for col in columns:
        if col not in df.columns:
            df[col] = default_value
    return df  # Optional: return for chaining


def extract_sentences_by_tuples(generated_tuples_filepath, sampled_pairs_filepath, sheet_name=None,
                                sampled_4_pairs_filepath=None):
    """
    Extract sentenceA and sentenceB from DataFrame based on ID tuples.
    Each row in output contains all 4 sentences with indexed prefixes.

    Parameters:
    - generated_tuples_filepath: path to file with ID tuples (4 IDs per line, tab-separated)
    - excel_file: path to Excel file with 'id', 'sentenceA', 'sentenceB' columns
    - sheet_name: Excel sheet name (None for first sheet)
    - sampled_4_pairs_filepath: path to save results (CSV or Excel)

    Returns:
    - DataFrame with columns: tuple_index, id_1-4, sentenceA_1-4, sentenceB_1-4
    """

    print("Reading tuples file...")
    # Read the tuples file
    with open(generated_tuples_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse ID tuples from each line
    id_tuples = []
    for line_num, line in enumerate(lines, 1):
        ids = line.strip().split('\t')
        if len(ids) >= 4:
            id_tuples.append(ids[:4])  # Take first 4 IDs
        else:
            print(f"Warning: Line {line_num} has only {len(ids)} IDs: {ids}")
            # Pad with empty strings if needed
            while len(ids) < 4:
                ids.append('')
            id_tuples.append(ids)

    print(f"Found {len(id_tuples)} tuples of IDs")

    # Read the Excel file
    print("Reading Excel file...")
    df = utils.load_dataframe(sampled_pairs_filepath, sheet_name=sheet_name)

    print(f"Excel DataFrame loaded with {len(df)} rows")
    print(f"Available columns: {list(df.columns)}")

    df.rename(columns={'SentenceA': 'SentenceA', 'SentenceB': 'SentenceB'}, inplace=True)

    # Create lookup dictionary for faster access
    print("Creating ID lookup dictionary...")
    id_lookup = {}
    for idx, row in df.iterrows():
        id_val = row['ID']
        id_lookup[id_val] = {
            'sentenceA': row['SentenceA'],
            'sentenceB': row['SentenceB']
        }

    print(f"Created lookup for {len(id_lookup)} unique IDs")

    # Process each tuple and create result rows
    result_rows = []
    missing_ids = set()

    for tuple_idx, id_tuple in enumerate(id_tuples, 1):
        row_data = {'tuple_index': tuple_idx}

        # Process each of the 4 IDs in the tuple
        for pos in range(4):
            id_val = id_tuple[pos] if pos < len(id_tuple) else ''

            # Add ID to row
            row_data[f'id_{pos + 1}'] = id_val

            # Look up sentences
            if id_val and id_val in id_lookup:
                row_data[f'pair_{pos + 1}_A'] = id_lookup[id_val]['sentenceA']
                row_data[f'pair_{pos + 1}_B'] = id_lookup[id_val]['sentenceB']
            elif id_val:
                # ID not found
                print(
                    f"\nWarning: In tuple_idx [{tuple_idx}], id_tuple [{id_tuple}], pos [{pos}] - id_val {id_val} was not found!")
                row_data[f'pair_{pos + 1}_A'] = 'NOT_FOUND'
                row_data[f'pair_{pos + 1}_B'] = 'NOT_FOUND'
                missing_ids.add(id_val)
            else:
                # Empty ID
                print(f"\nWarning: In tuple_idx [{tuple_idx}], id_tuple [{id_tuple}], pos [{pos}] - Empty id_val!")
                row_data[f'pair_{pos + 1}_A'] = ''
                row_data[f'pair_{pos + 1}_B'] = ''

        result_rows.append(row_data)

    # Create result DataFrame
    result_df = pd.DataFrame(result_rows)

    # Report missing IDs
    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} IDs not found in Excel file:")
        for missing_id in sorted(missing_ids):
            print(f"  - {missing_id}")

    # Save results if output file specified
    if sampled_4_pairs_filepath:
        utils.save_dataframe(result_df, sampled_4_pairs_filepath)
        print(f"Results saved successfully!")

    return result_df


def generate_bws_tuples(sampled_pairs_filepath, work_dir='./bws', temp_ids_filename='temp_ids.text',
                        sampled_4_pairs_filename='sampled_4_pairs.csv'):
    # Create the working dir if it doesn't exist.
    os.makedirs(work_dir, exist_ok=True)

    ids_filepath = os.path.join(work_dir, temp_ids_filename)
    sampled_4_pairs_filepath = os.path.join(work_dir, sampled_4_pairs_filename)

    df = utils.load_dataframe(sampled_pairs_filepath)
    id_list = df['ID'].tolist()

    # Save the list to a text file, one item per line
    with open(ids_filepath, 'w') as f:
        for item in id_list:
            f.write(str(item) + '\n')

    # perl generate-BWS-tuples.pl example-items-test1.txt
    tuples_script = os.path.join(BWS_DIR, 'generate-BWS-tuples.pl')
    cmd = ['perl', tuples_script, ids_filepath]
    subprocess.run(cmd)

    extract_sentences_by_tuples(ids_filepath + '.tuples', sampled_pairs_filepath, sheet_name=None,
                                sampled_4_pairs_filepath=sampled_4_pairs_filepath)


def calculate_bws_scores(selected_pairs_filepath, source_4_pair_annotations_filename, dest_pairs_scored_filename,
                         work_dir='./bws'):
    # Create the working dir if it doesn't exist.
    os.makedirs(work_dir, exist_ok=True)

    # Consts.
    first_cols = ['Item1', 'Item2', 'Item3', 'Item4', 'BestItem', 'WorstItem']
    dest_scored_pair_columns_mapping = {'Example in Sentence A': 'SentenceA', 'Example in Sentence B': 'SentenceB'}

    # Filenames.
    base_dest_pairs_scored_filename = os.path.splitext(dest_pairs_scored_filename)[0]
    dest_annotated_scored_csv_bws_formatted_filename = base_dest_pairs_scored_filename + '_bws_formatted.csv'
    dest_annotated_scored_csv_filename = base_dest_pairs_scored_filename + '_scored.tsv'

    # Paths.
    source_4_pair_annotations_filepath = os.path.join(work_dir, source_4_pair_annotations_filename)
    source_pairs_ready_4_scoring_csv_filepath = os.path.join(work_dir, dest_annotated_scored_csv_bws_formatted_filename)
    dest_annotated_scored_csv_filepath = os.path.join(work_dir, dest_annotated_scored_csv_filename)
    dest_pairs_scored_filepath = os.path.join(work_dir, dest_pairs_scored_filename)
    # Preparing a file that bws can annotate.
    source_annotated_df = utils.load_dataframe(source_4_pair_annotations_filepath)
    cols_to_rename = {"id_1": "Item1", "id_2": "Item2", "id_3": "Item3", "id_4": "Item4", "best_pair": "BestItem",
                      "worst_pair": "WorstItem"}
    source_annotated_df.rename(columns=cols_to_rename, inplace=True)
    add_missing_columns(source_annotated_df, first_cols)
    source_annotated_df = source_annotated_df[first_cols]
    source_annotated_df.to_csv(source_pairs_ready_4_scoring_csv_filepath, index=False)

    # get-scores-from-BWS-annotations-counting.pl
    bms_command = os.path.join(BWS_DIR, 'get-scores-from-BWS-annotations-counting.pl')
    cmd = ['perl', bms_command, source_pairs_ready_4_scoring_csv_filepath]

    with open(dest_annotated_scored_csv_filepath, 'w') as f:
        subprocess.run(cmd, stdout=f)

    # Load annotated scored data.
    dest_annotated_scored_df = pd.read_csv(dest_annotated_scored_csv_filepath, header=None, names=['ID', 'score'],
                                           sep='\t')
    dest_annotated_scored_df['score'] = (dest_annotated_scored_df['score'] + 1) / 2  # Normalizing.

    # Load selected pairs.
    selected_pairs_df = utils.load_dataframe(selected_pairs_filepath)
    selected_pairs_df.drop(columns=['cosine', 'bin', 'score'], inplace=True, errors='ignore')

    # Merging between the dataframes to enrich the annotations with the sentence pairs.
    merged_df = pd.merge(selected_pairs_df, dest_annotated_scored_df, on='ID', how='left')
    merged_df.rename(columns=dest_scored_pair_columns_mapping, inplace=True)

    # Saving.
    utils.save_dataframe(merged_df, dest_pairs_scored_filepath)
