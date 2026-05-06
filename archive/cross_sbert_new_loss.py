# from email import parser

# from datasets import load_dataset
# from angle_emb import AnglE, AngleDataTokenizer, CorrelationEvaluator
import torch.nn as nn
import argparse
# import collections

import pandas as pd
import numpy as np

import shutil
import os

from datasets import Dataset
from huggingface_hub import login
from pathlib import Path

from pandas.core.interchange.dataframe_protocol import DataFrame
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from tqdm import tqdm
from boltons.iterutils import chunked_iter
from sklearn.metrics.pairwise import paired_cosine_distances

import model_utils
import utils
import text_utils
from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder, evaluation
from torch.utils.data import DataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Should be done when running.
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XET_NO_PROGRESS"] = "true"

# Disable dataset progress bars
from datasets import disable_progress_bars

disable_progress_bars()


def main(args_list=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate sentence similarity')

    parser.add_argument('--data_dir', default='./data', help='The directory that stores the data (defaults to ./data)')
    parser.add_argument('--train_dir', default='./data', help='The directory that stores the train data (defaults to ./data)')
    parser.add_argument('--model_dir', default='ckpts/sts-b', help='The directory that stores the output model')
    parser.add_argument('--results_dir', default='./results/4-12-all-runs', help='The directory that stores the results (defaults to ./results)')

    parser.add_argument('--train_filename', default='train_pairs_B_shuffled_600_scored.xlsx', help='The filename containing the training data')
    parser.add_argument('--train2_filename', default=None, help='Optional second training file')
    parser.add_argument('--validation_filename', default='validation_pairs_B_shuffled_100_scored.xlsx', help='The filename containing the validation data')
    parser.add_argument('--test_filename', default='test_pairs_B_shuffled_300_scored.xlsx', help='The filename containing the test data')
    parser.add_argument('--test2_filename', default=None, help='Optional second test file')

    parser.add_argument('--hf_base_model', default='OMRIDRORI/mbert-tibetan-continual-wylie-final', help='The Hugging Face base model identifier name')
    parser.add_argument('--hf_token', default=None, help='The Hugging Face access token')

    parser.add_argument('--results_filename', default='B-Results.csv', help='The filename to save the results')
    parser.add_argument('--keep_previous_model_in_dir', action='store_true', help='When marked the previous model with the check points will not be deleted')
    parser.add_argument('--seed', type=int, default=42, help='the random seed (defaults to 0)')

    parser.add_argument('--no_fit', default=False, action='store_true', help='Do not do any fitting of the model')

    parser.add_argument('--pool_of_pairs_filename', help='The given file will be added a cosine column with the similarity score')
    parser.add_argument('--pool_of_pairs_cosine_filename', help='If pool_of_pairs_filename was given the result will output to this file')

    pretrained_config_group = parser.add_argument_group('Additional for pretrained_config','Additional for pretrained_config')
    pretrained_config_group.add_argument('--pretrained_model_path', type=str, default=None,help='pretrained_model_path (defaults to None)')
    pretrained_config_group.add_argument('--pretrained_lora_path', type=str, default=None, help='pretrained_lora_path (defaults to None)')
    pretrained_config_group.add_argument('--is_llm', action='store_true', help='Set this flag if the model is an LLM (defaults to False)')
    pretrained_config_group.add_argument('--pooling_strategy', type=str, default='cls', help='pooling_strategy (defaults to cls)')
    pretrained_config_group.add_argument('--train_mode', type=bool, default=False, help='train_mode (defaults to False)')
    pretrained_config_group.add_argument('--local_files_only', type=bool, default=False, help='If True will load the files from local path')

    fit_config_group = parser.add_argument_group('Additional for fit_config', 'Additional for fit_config')
    fit_config_group.add_argument('--batch_size', type=int, default=32, help='batch_size (defaults to 32)')
    fit_config_group.add_argument('--epochs', type=int, default=5, help='epochs (defaults to 5)')
    fit_config_group.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate (defaults to 2e-5)')
    fit_config_group.add_argument('--save_steps', type=int, default=100, help='save_steps (defaults to 100)')
    fit_config_group.add_argument('--eval_steps', type=int, default=1000, help='eval_steps (defaults to 1000)')
    fit_config_group.add_argument('--warmup_steps', type=int, default=0, help='warmup_steps (defaults to 0)')
    fit_config_group.add_argument('--gradient_accumulation_steps', type=int, default=16, help='gradient_accumulation_steps (defaults to 16)')
    fit_config_group.add_argument('--fp16', action='store_true', default=True, help='fp16 (defaults to True)')
    fit_config_group.add_argument('--logging_steps', type=int, default=100, help='logging_steps (defaults to 100)')

    args = parser.parse_args(args_list)

    # --- Setup Configuration Dictionaries ---
    df_mapping = {'SentenceA': 'text1', 'SentenceB': 'text2', 'score': 'label'}
    datasets_config = {
        'train_filename': args.train_filename,
        'train2_filename': args.train2_filename,
        'validation_filename': args.validation_filename,
        'test_filename': args.test_filename,
        'test2_filename': args.test2_filename,
        'random_state': args.seed,
    }

    model_name_or_path = args.hf_base_model
    pretrained_model_path = args.pretrained_model_path
    if args.pretrained_model_path:
        pretrained_model_path = Path(args.pretrained_model_path)

    pretrained_config = {
        'model_name_or_path': model_name_or_path,
        'pretrained_model_path': pretrained_model_path,
        'pretrained_lora_path': args.pretrained_lora_path,
        'is_llm': args.is_llm,
        'pooling_strategy': args.pooling_strategy,
        'train_mode': args.train_mode,
        'local_files_only': args.local_files_only,
    }

    fit_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'warmup_steps': args.warmup_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'fp16': args.fp16,
        'logging_steps': args.logging_steps,
    }

    # Initialize run_settings once
    run_settings = {"model": "Cross-Encoder-SBERT", "manual": "1", "no_fit": args.no_fit}
    run_settings.update(datasets_config)
    run_settings.update(pretrained_config)
    run_settings.update(fit_config)

    # --- Initial Setup ---
    if args.hf_token:
        login(token=args.hf_token)

    # getting the datasets
    train_dataset, train2_dataset, validation_dataset, test_dataset, test2_dataset = text_utils.get_datasets(
        datasets_config, args.data_dir, args.train_dir, df_mapping)
    print('Datasets loaded')

    # load pretrained model
    model = CrossEncoder(model_name_or_path, num_labels=1)
    print(f'Model loaded: {model_name_or_path}')

    train_ds = train_dataset
    train2_ds = train2_dataset
    valid_ds = validation_dataset
    test_ds = test_dataset
    test2_ds = test2_dataset
    print('Datasets tokenized')

    # Training the model.

    # Delete checkpoint folder if it exists
    if (not args.keep_previous_model_in_dir) and os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
        print('Deleted previous model!')
    else:
        print('Previous model was kept. (if such exists)')

    # Ensure results directory exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        print(f"Results folder created:{args.results_dir}")

    log_results_file = os.path.join(args.results_dir, args.results_filename)

    # --- FIRST FIT (Train 1 / Synthetic Set) ---
    if not args.no_fit:
        print("Starting fine-tuning (Train 1) with CrossEncoder...")

        # Filter out rows where text1 or text2 are empty/invalid.
        filtered_train_ds = [
            row for row in train_ds
            if isinstance(row.get('text1'), str) and row['text1'].strip() and
               isinstance(row.get('text2'), str) and row['text2'].strip()
        ]
        if len(filtered_train_ds) < len(train_ds):
            print(
                f"Warning: Removed {len(train_ds) - len(filtered_train_ds)} rows with empty/invalid text inputs in Train 1.")

        # Convert to SBERT InputExamples
        train_samples = [
            InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
            for row in filtered_train_ds
        ]

        # Create DataLoader
        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=fit_config['batch_size']
        )

        # Run fine-tuning
        model.fit(
            train_dataloader,
            loss_fct=nn.BCEWithLogitsLoss(),  # Use MSELoss for regression scores
            epochs=fit_config['epochs'],
            warmup_steps=fit_config['warmup_steps'],
            output_path=args.model_dir,
            show_progress_bar=True,
            optimizer_params={'lr': fit_config['learning_rate']},
        )
        print("Fine-tuning (Train 1) complete.")

        # --- EVALUATION (AFTER FIRST FIT) ---
        print('evaluating after Train 1 (Intermediate Metrics)...')

        train_metrics = model_utils.cross_encoder_evaluate(model, train_ds, batch_size=4)
        print('Train metrics (Intermediate):', train_metrics)

        validation_metrics = model_utils.cross_encoder_evaluate(model, valid_ds, batch_size=4)
        print('Validation metrics (Intermediate):', validation_metrics)

        test_metrics = model_utils.cross_encoder_evaluate(model, test_ds, batch_size=4)
        print('Test metrics (Intermediate):', test_metrics)

        test2_metrics = None
        if test2_ds is not None:
            print("Evaluating SECOND test set (Intermediate)...")
            test2_metrics = model_utils.cross_encoder_evaluate(model, test2_ds, batch_size=4)
            print("Test2 metrics (Intermediate):", test2_metrics)

        print('Done intermediate evaluating.')

        # --- LOG RESULTS AFTER FIRST FIT (Log line 1) ---
        # Use empty prefix ("") for standard column names (e.g., train_spearman)
        eval_result_1 = model_utils.create_eval_result(
            train_metrics, validation_metrics, test_metrics, test2_metrics, prefix=""
        )

        # Log to the final results file (This is the first line, append_mode=False is default/overwrite)
        utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result_1)
        print(f"First fit results logged (overwritten) to: {log_results_file}")

    else:
        print("Skipping fine-tuning (--no_fit enabled).")

    # --- SECOND FIT (Train 2) ---
    if not args.no_fit and train2_ds is not None:
        print("Starting second fine-tuning (Train 2) with CrossEncoder...")

        # Filter out rows where text1 or text2 are empty/invalid.
        filtered_train2_ds = [
            row for row in train2_ds
            if isinstance(row.get('text1'), str) and row['text1'].strip() and
               isinstance(row.get('text2'), str) and row['text2'].strip()
        ]
        if len(filtered_train2_ds) < len(train2_ds):
            print(
                f"Warning: Removed {len(train2_ds) - len(filtered_train2_ds)} rows with empty/invalid text inputs in Train 2.")

        # Convert train2 dataset rows into SBERT InputExamples
        train2_samples = [
            InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
            for row in filtered_train2_ds
        ]

        # Create DataLoader
        train2_dataloader = DataLoader(
            train2_samples,
            shuffle=True,
            batch_size=fit_config['batch_size']
        )

        # Run fine-tuning on the already trained model
        model.fit(
            train2_dataloader,
            loss_fct=nn.MSELoss(),
            epochs=fit_config['epochs'],
            warmup_steps=fit_config['warmup_steps'],
            output_path=args.model_dir,
            show_progress_bar=True,
            optimizer_params={'lr': fit_config['learning_rate']},
        )
        print("Fine-tuning (Train 2) complete.")

        # --- EVALUATION (AFTER SECOND FIT) ---
        print('evaluating final model performance...')

        train_metrics_2 = model_utils.cross_encoder_evaluate(model, train_ds, batch_size=4)
        print('Train set pearson (FINAL):', train_metrics_2['pearson_cosine'])

        validation_metrics_2 = model_utils.cross_encoder_evaluate(model, valid_ds, batch_size=4)
        print('Validation set pearson (FINAL):', validation_metrics_2['pearson_cosine'])

        test_metrics_2 = model_utils.cross_encoder_evaluate(model, test_ds, batch_size=4)
        print('Test set pearson (FINAL):', test_metrics_2['pearson_cosine'])

        test2_metrics_2 = None
        if test2_ds is not None:
            print("Evaluating SECOND test set (FINAL)...")
            test2_metrics_2 = model_utils.cross_encoder_evaluate(model, test2_ds, batch_size=4)
            print("Test2 set pearson (FINAL):", test2_metrics_2['pearson_cosine'])

        print('Done final evaluating')

        # --- LOG RESULTS AFTER SECOND FIT (Log line 2) ---
        # Use prefix "2" to generate column names like "train2spearman"
        eval_result_2 = model_utils.create_eval_result(
            train_metrics_2, validation_metrics_2, test_metrics_2, test2_metrics_2, prefix="second_fit_"
        )

        # Log results to the same file, setting append_mode=True
        utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result_2)
        print(f"Second fit results logged to: {log_results_file}")

    # Evaluating pairs Cosine values to candidate_pairs
    if args.pool_of_pairs_filename:
        # Note: model_utils.calculate_pairs_cosine is designed for bi-encoders.
        # For cross-encoders, this function might fail or produce meaningless cosine results.
        # Assuming model_utils.calculate_pairs_cosine is capable of handling the CrossEncoder model object.
        model_utils.calculate_pairs_cosine(model, os.path.join(args.results_dir, args.pool_of_pairs_filename),
                                           os.path.join(args.results_dir,
                                                        args.pool_of_pairs_cosine_filename or args.pool_of_pairs_filename))

    # Delete checkpoint folder if it exists
    if (not args.keep_previous_model_in_dir) and os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
        print('Deleted previous model!')
    else:
        print('Previous model was kept. (if such exists)')


if __name__ == '__main__':
    main()