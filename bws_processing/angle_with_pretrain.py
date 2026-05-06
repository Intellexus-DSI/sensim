# angle.py (Final Version)
from angle_emb import AnglE, AngleDataTokenizer, evaluation

import argparse
import shutil
import os
import gc
import torch
import json  # ADDED: For reading trainer_state.json
import math  # ADDED: For ceiling calculation

from huggingface_hub import login

import text_utils
import model_utils_pretrain
import utils

# os.environ["WANDB_DISABLED"] = "true"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["XET_NO_PROGRESS"] = "true"

#os.environ["WANDB_API_KEY"] = ""
import wandb
wandb.init(project="AnglE")

from transformers import logging as hf_logging

# To see everything (progress bars, config info)
hf_logging.set_verbosity_info()

from datasets import disable_progress_bars

disable_progress_bars()


# =================================================================
# NEW UTILITY FUNCTION: Find the best checkpoint path and step
# =================================================================
def find_best_checkpoint(output_dir):
    """
    Reads the trainer_state.json to find the path and step of the best checkpoint.
    Returns the absolute path to the checkpoint directory and the step number.
    """
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        return None, None

    try:
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)

        best_model_checkpoint = state.get("best_model_checkpoint")

        best_step = None
        if best_model_checkpoint:
            # Construct the absolute path to the best checkpoint subdirectory
            best_checkpoint_path = os.path.join(output_dir, best_model_checkpoint)

            # Extract step number from the path (e.g., '.../checkpoint-1200' -> 1200)
            try:
                best_step = int(best_model_checkpoint.split('-')[-1])
            except ValueError:
                pass

            return best_checkpoint_path, best_step
    except Exception as e:
        print(f"Error reading trainer_state.json: {e}")

    return None, None


# =================================================================


def main(args_list=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate sentence similarity')
    # ... (parser arguments remain the same) ...
    parser.add_argument('--data_dir', default='./data/B',
                        help='The directory that stores the data (defaults to ./data)')
    parser.add_argument('--train_dir', default='./data/B',
                        help='The directory that stores the train data (defaults to ./data)')
    parser.add_argument('--model_dir', default='ckpts/sts-b', help='The directory that stores the output model')
    parser.add_argument('--results_dir', default='./results/4-12-all-runs',
                        help='The directory that stores the results (defaults to ./results)')

    parser.add_argument('--train_filename', default='train_pairs_B_shuffled_600_scored.xlsx',
                        help='The filename containing the training data')
    parser.add_argument('--train2_filename', default=None, help='Optional second training file')
    parser.add_argument('--validation_filename', default='validation_pairs_B_shuffled_150_scored.xlsx',
                        help='The filename containing the validation data')
    parser.add_argument('--test_filename', default='test_pairs_B_shuffled_250_scored.xlsx',
                        help='The filename containing the test data')
    parser.add_argument('--test2_filename', default=None, help='Optional second test file')

    parser.add_argument('--hf_base_model', default='OMRIDRORI/mbert-tibetan-continual-wylie-final',
                        help='The Hugging Face base model identifier name')
    parser.add_argument('--hf_token', default=None,
                        help='The Hugging Face access token')

    parser.add_argument('--results_filename', default='B-Results.csv', help='The filename to save the results')
    parser.add_argument('--keep_previous_model_in_dir', action='store_true',
                        help='When marked the previous model with the check points will not be deleted')
    parser.add_argument('--seed', type=int, default=42, help='the random seed (defaults to 0)')

    parser.add_argument('--no_fit', default=False, action='store_true', help='Do not do any fitting of the model')

    parser.add_argument('--pool_of_pairs_filename',
                        help='The given file will be added a cosine column with the similarity score')
    parser.add_argument('--pool_of_pairs_cosine_filename',
                        help='If pool_of_pairs_filename was given the result will output to this file')


    pretrained_config_group = parser.add_argument_group('Additional for pretrained_config',
                                                        'Additional for pretrained_config')
    pretrained_config_group.add_argument('--pretrained_model_path', type=str, default=None,
                                         help='pretrained_model_path (defaults to None)')
    pretrained_config_group.add_argument('--pretrained_lora_path', type=str, default=None,
                                         help='pretrained_lora_path (defaults to None)')
    pretrained_config_group.add_argument('--is_llm', action='store_true',
                                         help='Set this flag if the model is an LLM (defaults to False)')
    pretrained_config_group.add_argument('--pooling_strategy', type=str, default='cls',
                                         help='pooling_strategy (defaults to cls)')
    pretrained_config_group.add_argument('--train_mode', type=bool, default=False,
                                         help='train_mode (defaults to False)')
    pretrained_config_group.add_argument('--local_files_only', type=bool, default=False,
                                         help='If True will load the files from local path')

    loss_kwargs_group = parser.add_argument_group('Additional for loss_kwargs', 'Additional for loss_kwargs')
    loss_kwargs_group.add_argument('--cosine_w', type=float, default=1e-2, help='cosine_w (defaults to 1e-2)')
    loss_kwargs_group.add_argument('--ibn_w', type=float, default=1.0, help='ibn_w (defaults to 1.0)')
    loss_kwargs_group.add_argument('--cln_w', type=float, default=1.0, help='cln_w (defaults to 1.0)')
    loss_kwargs_group.add_argument('--angle_w', type=float, default=0.02, help='angle_w (defaults to 0.02)')
    loss_kwargs_group.add_argument('--cosine_tau', type=float, default=20, help='cosine_tau (defaults to 20)')
    loss_kwargs_group.add_argument('--ibn_tau', type=float, default=20, help='ibn_tau (defaults to 20)')
    loss_kwargs_group.add_argument('--angle_tau', type=float, default=20, help='angle_tau (defaults to 20)')

    fit_config_group = parser.add_argument_group('Additional for fit_config', 'Additional for fit_config')
    fit_config_group.add_argument('--batch_size', type=int, default=32, help='batch_size (defaults to 32)')
    fit_config_group.add_argument('--epochs', type=int, default=5, help='epochs (defaults to 5)')
    fit_config_group.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate (defaults to 2e-5)')
    fit_config_group.add_argument('--save_steps', type=int, default=300, help='save_steps (defaults to 100)')
    fit_config_group.add_argument('--eval_steps', type=int, default=300, help='eval_steps (defaults to 1000)')
    fit_config_group.add_argument('--warmup_steps', type=int, default=0, help='warmup_steps (defaults to 0)')
    fit_config_group.add_argument('--gradient_accumulation_steps', type=int, default=16,
                                  help='gradient_accumulation_steps (defaults to 16)')
    fit_config_group.add_argument('--fp16', action='store_true', default=True, help='fp16 (defaults to True)')
    fit_config_group.add_argument('--logging_steps', type=int, default=300, help='logging_steps (defaults to 100)')
    # ... (end of parser arguments) ...
    # CRITICAL FIX: Add evaluation strategy argument to parser
    fit_config_group.add_argument('--evaluation_strategy', type=str, default='steps',
                                  help='Evaluation strategy for HuggingFace Trainer (e.g., steps, epoch, no)')
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

    pretrained_config = {
        'model_name_or_path': model_name_or_path,
        'pretrained_model_path': pretrained_model_path,
        'pretrained_lora_path': args.pretrained_lora_path,
        'is_llm': args.is_llm,
        'pooling_strategy': args.pooling_strategy,
        'train_mode': args.train_mode,
        'local_files_only': args.local_files_only,
    }

    loss_kwargs = {
        'cosine_w': args.cosine_w,
        'ibn_w': args.ibn_w,
        'cln_w': args.cln_w,
        'angle_w': args.angle_w,
        'cosine_tau': args.cosine_tau,
        'ibn_tau': args.ibn_tau,
        'angle_tau': args.angle_tau
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
        'eval_strategy': args.evaluation_strategy,
        #'save_strategy': 'best',
        #'save_total_limit': 3,
    }

    run_settings = {"model": "AnglE", "manual": "1", "no_fit": args.no_fit}
    run_settings.update(datasets_config)
    run_settings.update(pretrained_config)
    run_settings.update(loss_kwargs)
    run_settings.update(fit_config)

    # --- Initial Setup ---
    unified_pretrained_config = pretrained_config
    if args.hf_token:
        login(token=args.hf_token)
        unified_pretrained_config = unified_pretrained_config | {"token": args.hf_token}

        # getting the datasets
    train_dataset, train2_dataset, validation_dataset, test_dataset, test2_dataset = text_utils.get_datasets(
        datasets_config, args.data_dir, args.train_dir, df_mapping)
    print('Datasets loaded')

    # load pretrained model
    angle = AnglE.from_pretrained(**unified_pretrained_config).cuda()
    print('Model loaded')

    # Tokenizing data.
    angle_tokenizer = AngleDataTokenizer(angle.tokenizer, angle.max_length)

    # Tokenizing data.
    train_ds = train_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    valid_ds = validation_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    test_ds = test_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)

    train2_ds = None
    if train2_dataset is not None:
        print("Tokenizing second training set...")
        train2_ds = train2_dataset.map(angle_tokenizer, num_proc=8)

    test2_ds = None
    if test2_dataset is not None:
        if 'label' in test2_dataset.column_names:
            test2_ds = test2_dataset.map(angle_tokenizer, num_proc=8)
        else:
            print("⚠ Test2 has no labels — skipping tokenization")
            test2_ds = test2_dataset

    print('Datasets tokenized')

    # =================================================================
    # NEW: Calculate steps per epoch for epoch tracking (First Fit)
    # =================================================================
    train_dataset_size = len(train_ds)
    batch_size_effective = fit_config['batch_size'] * fit_config['gradient_accumulation_steps']

    if batch_size_effective == 0:
        steps_per_epoch = 1
    else:
        steps_per_epoch = math.ceil(train_dataset_size / batch_size_effective)

    print(f"Calculated steps per epoch for First Fit: {steps_per_epoch}")

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

    # Placeholder for the best checkpoint path and step from the first fit
    best_checkpoint_path_1 = None
    best_step_1 = None


    #evaluator=evaluation.CorrelationEvaluator(text1=valid_ds['text1'],text2=valid_ds['text2'],labels=valid_ds['label'], batch_size=32)

    argument_kwargs = {
        #'metric_for_best_model': 'eval_spearman_cosine',
        'log_level': 'debug',
        #metric_for_best_model = 'eval_spearman_cosine'
        #'evaluator':evaluator,
    }

    # --- FIRST FIT (Train 1 / Synthetic Set) ---
    if not args.no_fit:
        angle.fit(
            train_ds=train_ds,
            valid_ds=valid_ds,
            valid_ds_for_callback=valid_ds,
            output_dir=args.model_dir,
            loss_kwargs=loss_kwargs,
            argument_kwargs=argument_kwargs,

            **fit_config
        )
        print('done fitting (Train 1')

        best_checkpoint_path = os.path.join(args.model_dir, "best-checkpoint")
        if os.path.exists(best_checkpoint_path):
            angle = AnglE.from_pretrained(best_checkpoint_path).cuda()
            print(f"Reloaded AnglE model from BEST CHECKPOINT ({best_checkpoint_path}) for evaluation.")
        else:
            print(f"Best checkpoint path does not exist: {best_checkpoint_path}. Loading from model_dir root.")
            angle = AnglE.from_pretrained(args.model_dir).cuda()

        # --- EVALUATION (AFTER FIRST FIT) ---
        print('evaluating after Train 1 ...')

        train_metrics = model_utils_pretrain.bi_encoder_evaluate(angle, train_ds, batch_size=4)
        validation_metrics = model_utils_pretrain.bi_encoder_evaluate(angle, valid_ds, batch_size=4)
        test_metrics = model_utils_pretrain.bi_encoder_evaluate(angle, test_ds, batch_size=4)

        test2_metrics = None
        if test2_ds is not None and 'label' in test2_ds.column_names:
            test2_metrics = model_utils_pretrain.bi_encoder_evaluate(angle, test2_ds, batch_size=4)

        print('Done evaluating.')

        # --- LOG RESULTS AFTER FIRST FIT (Log line 1 - Best Checkpoint Metrics) ---
        eval_result_1 = model_utils_pretrain.create_eval_result(
            train_metrics, validation_metrics, test_metrics, test2_metrics, prefix=""
        )


        utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result_1)
        print(f"First fit results (Best Checkpoint) logged to: {log_results_file}")

    # Clear so memory is available for second fit
    gc.collect()
    torch.cuda.empty_cache()



    # --- SECOND FIT (Train 2) ---
    if not args.no_fit and train2_ds is not None:
        print("Starting second fit with train2 dataset...")

        model_dir = os.path.join(args.model_dir, "second_fit")

        angle.fit(
            train_ds=train2_ds,
            valid_ds=valid_ds,
            valid_ds_for_callback=valid_ds,
            output_dir=model_dir,
            loss_kwargs=loss_kwargs,
            argument_kwargs=argument_kwargs,
            **fit_config
        )
        print('done with second fitting (Train 2)')

        best_checkpoint_path = os.path.join(model_dir, "best-checkpoint")
        if os.path.exists(best_checkpoint_path):
            angle = AnglE.from_pretrained(best_checkpoint_path).cuda()
            print(f"Reloaded AnglE model from BEST CHECKPOINT ({best_checkpoint_path}) for evaluation.")
        else:
            print(f"Best checkpoint path does not exist: {best_checkpoint_path}. NOT RELOADING model.")


        # --- EVALUATION (AFTER SECOND FIT) ---
        print('evaluating final model performance...')

        train_metrics_2 = model_utils_pretrain.bi_encoder_evaluate(angle, train_ds, batch_size=4)
        validation_metrics_2 = model_utils_pretrain.bi_encoder_evaluate(angle, valid_ds, batch_size=4)
        test_metrics_2 = model_utils_pretrain.bi_encoder_evaluate(angle, test_ds, batch_size=4)

        test2_metrics_2 = None
        if test2_ds is not None and 'label' in test2_ds.column_names:
            test2_metrics_2 = model_utils_pretrain.bi_encoder_evaluate(angle, test2_ds, batch_size=4)

        print('Done final evaluating')

        # --- LOG RESULTS AFTER SECOND FIT (Log line 2) ---
        eval_result_2 = model_utils_pretrain.create_eval_result(
            train_metrics_2, validation_metrics_2, test_metrics_2, test2_metrics_2, prefix="second_fit_"
        )

        utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result_2)
        print(f"Second fit results logged to: {log_results_file}")

    # --- Cleanup ---
    if args.pool_of_pairs_filename:
        model_utils_pretrain.calculate_pairs_cosine(angle, os.path.join(args.results_dir, args.pool_of_pairs_filename),
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