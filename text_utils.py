import pandas as pd
import numpy as np

import shutil
import os

import utils
from datasets import Dataset # Added to ensure Dataset type is handled, although not strictly needed for this file

def clean_sentences(df: pd.DataFrame) -> pd.DataFrame:
    df['SentenceA'] = df['SentenceA'].str.replace(r'\t+', ' ', regex=True)
    df['SentenceB'] = df['SentenceB'].str.replace(r'\t+', ' ', regex=True)

    df = df[df['SentenceA'].notna() & (df['SentenceA'].str.strip().str.len() > 0)]
    df = df[df['SentenceB'].notna() & (df['SentenceB'].str.strip().str.len() > 0)]

    return df


def get_datasets(datasets_config, data_dir, train_dir, df_mapping):
    """
    Loads up to 5 datasets: train (1), train2 (optional), validation, test (1), and test2 (optional).
    """
    random_state = datasets_config['random_state']

    # 1. Load Training Dataset 1
    train_dataset = utils.get_dataset(os.path.join(train_dir, datasets_config['train_filename']), df_mapping, random_state)

    # 2. Load Optional Training Dataset 2 (REQUIRED FIX)
    train2_dataset = None
    if datasets_config.get("train2_filename"):
        train2_path = os.path.join(train_dir, datasets_config['train2_filename'])
        if os.path.exists(train2_path):
            train2_dataset = utils.get_dataset(train2_path, df_mapping, random_state)

    # 3. Load Validation Dataset
    validation_dataset = utils.get_dataset(os.path.join(data_dir, datasets_config['validation_filename']), df_mapping, random_state)

    # 4. Load Test Dataset 1
    test_dataset = utils.get_dataset(os.path.join(data_dir, datasets_config['test_filename']), df_mapping, random_state)

    # 5. Load Optional Test Dataset 2
    test2_dataset = None
    if datasets_config.get("test2_filename"):
        test2_path = os.path.join(data_dir, datasets_config['test2_filename'])
        if os.path.exists(test2_path):
            test2_dataset = utils.get_dataset(test2_path, df_mapping, random_state)

    # Return 5 datasets, including train2_dataset
    return train_dataset, train2_dataset, validation_dataset, test_dataset, test2_dataset


def get_multiple_datasets(train_fils, valid_file, test_files, data_dir, train_dir, df_mapping, eval_df_mapping=None, random_state=42):

    if eval_df_mapping is None:
        eval_df_mapping = df_mapping

    # 1. Load Training Datasets
    train_datasets = []
    for train_file in train_fils:
        train_path = os.path.join(train_dir, train_file)
        train_dataset = utils.get_dataset(train_path, df_mapping, random_state)
        train_datasets.append(train_dataset)


    # 3. Load Validation Dataset
    if valid_file is None:
        validation_dataset = None
    else:
        validation_dataset = utils.get_dataset(os.path.join(data_dir, valid_file), eval_df_mapping, random_state)

    # 4. Load Test Datasets
    test_datasets = []
    for test_file in test_files:
        test_path = os.path.join(data_dir, test_file)
        test_dataset = utils.get_dataset(test_path, eval_df_mapping, random_state)
        test_datasets.append(test_dataset)

    # Return ALL datasets.
    return train_datasets, validation_dataset, test_datasets