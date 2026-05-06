import argparse
import shutil
import os
import math
from pathlib import Path

from huggingface_hub import login
from sentence_transformers import (
    CrossEncoder,
    InputExample,
    evaluation,
)
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from transformers import TrainerCallback

import model_utils
import model_utils_pretrain
import text_utils
import utils
from app_config import AppConfig
from model_utils import calc_cosine
import weave

import torch

# added the following to enable ModernBERT architecture.
# 1. Force Torch Dynamo to ignore errors and fall back to "eager" mode
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# 2. Disable the Torch compiler globally
os.environ["TORCHDYNAMO_DISABLE"] = "1"


gpu_device = model_utils.get_gpu_device()

from datasets import disable_progress_bars

disable_progress_bars()


class _PerEvalCallback(TrainerCallback):
    """Evaluates the cross-encoder after each eval event and appends results to a CSV.

    When use_steps=True logs global_step; otherwise logs epoch number.
    """

    def __init__(self, model, train_ds, valid_ds, test_datasets, run_settings,
                 datasets_config, log_file, use_steps: bool = False):
        self._model = model
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._test_datasets = test_datasets
        self._run_settings = dict(run_settings)
        self._datasets_config = dict(datasets_config)
        self._log_file = log_file
        self._use_steps = use_steps

    def on_evaluate(self, args, state, control, **kwargs):
        if self._use_steps:
            step = state.global_step
            epoch = round(state.epoch, 2)
            eval_datasets_config = {
                **self._datasets_config,
                'evaluated_model': f'step_{step}',
                'step': step,
                'epoch': epoch,
            }
        else:
            epoch = int(round(state.epoch))
            eval_datasets_config = {
                **self._datasets_config,
                'evaluated_model': f'epoch_{epoch}',
                'epoch': epoch,
            }
        evaluate_and_log(
            self._model, self._train_ds, self._valid_ds, self._test_datasets,
            dict(self._run_settings),
            eval_datasets_config,
            self._log_file,
        )


def main(args_list=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate sentence similarity')
    # ... (parser arguments remain the same) ...
    parser.add_argument('--data_dir', default='./data', help='The directory that stores the data (defaults to ./data)')
    parser.add_argument('--train_dir', default='./data',
                        help='The directory that stores the train data (defaults to ./data)')
    parser.add_argument('--model_dir', default='ckpts/sts-b', help='The directory that stores the output model')
    parser.add_argument('--final_model_dir', default=None,
                        help='If set, save the final trained model to this directory after all training phases complete')
    parser.add_argument('--results_dir', default='./results/Cross-Trainer',
                        help='The directory that stores the results (defaults to ./results)')

    parser.add_argument('--train_filenames', nargs='+', default='train_pairs_B_shuffled_600_scored.xlsx llms_pairs_B_shuffled_2500_scored.xlsx',
                        help='The filenames containing the training data')

    parser.add_argument('--validation_filename', default='validation_pairs_B_shuffled_150_scored.xlsx',
                        help='The filename containing the validation data')

    parser.add_argument('--test_filenames', nargs='+', default='test_pairs_B_shuffled_250_scored.xlsx test_pairs_B_shuffled_no_positives_scored.xlsx',
                        help='The filenames containing the test data')

    parser.add_argument('--hf_base_model', default='Intellexus/mbert-tibetan-continual-wylie-final',
                        help='The Hugging Face base model identifier name')
    parser.add_argument('--hf_token', help='The Hugging Face access token')

    parser.add_argument('--results_filename', default='Cros-Results.csv', help='The filename to save the results')
    parser.add_argument('--keep_previous_model_in_dir', action='store_true',
                        help='When marked the previous model with the check points will not be deleted')
    parser.add_argument('--seed', type=int, default=42, help='the random seed (defaults to 0)')
    parser.add_argument('--use_unicode_columns', default=False, action='store_true',
                        help='Use SentenceA_unicode/SentenceB_unicode columns instead of SentenceA/SentenceB')

    parser.add_argument('--no_fit', default=False, action='store_true', help='Do not do any fitting of the model')

    parser.add_argument('--pool_of_pairs_filename',
                        help='The given file will be added a cosine column with the similarity score')
    parser.add_argument('--pool_of_pairs_cosine_filename',
                        help='If pool_of_pairs_filename was given the result will output to this file')

    pretrained_config_group = parser.add_argument_group('Additional for pretrained_config', 'Additional for pretrained_config')
    pretrained_config_group.add_argument('--pooling_strategy', type=str, default='weightedmean', choices=['cls', 'mean', 'lasttoken', 'max', 'mean_sqrt_len_tokens', 'weightedmean'],
                                         help='pooling_strategy (defaults to mean)')
    # pretrained_config_group.add_argument('--train_mode', type=bool, default=False,
    #                                      help='train_mode (defaults to False)')
    # pretrained_config_group.add_argument('--local_files_only', type=bool, default=False,
    #                                      help='If True will load the files from local path')

    fit_config_group = parser.add_argument_group('Additional for fit_config', 'Additional for fit_config')
    fit_config_group.add_argument('--batch_size', type=int, default=32, help='batch_size (defaults to 32)')
    fit_config_group.add_argument('--epochs', type=int, default=5, help='epochs (defaults to 5)')
    fit_config_group.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate (defaults to 2e-5)')
    fit_config_group.add_argument('--save_strategy', type=str, default="best", help='save_strategy [no, steps, epoch, best] (defaults to best)')
    fit_config_group.add_argument('--warmup_steps', type=int, default=100, help='warmup_steps (defaults to 0)')
    fit_config_group.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay (defaults to 0.1)')
    fit_config_group.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps (defaults to 1)')
    fit_config_group.add_argument('--loss_type', type=str, default='binaryCrossEntropyLoss', choices=['binaryCrossEntropyLoss', 'mseLoss'], help='Loss function to use: cosine (CosineSimilarityLoss) or cosent (CoSENTLoss)')
    fit_config_group.add_argument('--lr_scheduler_type', type=str, default='reduce_lr_on_plateau', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'cosine_warmup_with_min_lr', 'warmup_stable_decay'], help='The scheduler type to use')
    fit_config_group.add_argument('--eval_steps', type=int, default=None, help='eval_steps: if set, eval_strategy will be set to steps (defaults to None)')
    fit_config_group.add_argument('--no_per_epoch_eval', default=False, action='store_true', help='Disable per-epoch evaluation logging to the _epochs CSV')

    args = parser.parse_args(args_list)

    # Initialize env.
    initialize(args)

    load_best_model_at_end = args.save_strategy == 'best'
    model_name_or_path = args.hf_base_model

    train_filenames = args.train_filenames
    if isinstance(args.train_filenames, list):
        train_filenames = " ".join(args.train_filenames)
    train_filenames = train_filenames.split() if isinstance(train_filenames, str) else train_filenames

    test_filenames = args.test_filenames
    if isinstance(args.test_filenames, list):
        test_filenames = " ".join(args.test_filenames)
    test_filenames = test_filenames.split() if isinstance(test_filenames, str) else test_filenames

    datasets_config = {
        'validation_filename': args.validation_filename,
        'random_state': args.seed,
        'use_unicode_columns': args.use_unicode_columns,
    }

    for i, test_filename in enumerate(test_filenames):
        datasets_config[f'test{i}_filename'] = test_filename


    pretrained_config = {
        'model_name_or_path': model_name_or_path,
        #'train_mode': args.train_mode,
        #'local_files_only': args.local_files_only,
        'pooling_strategy': args.pooling_strategy,
        'loss_type': args.loss_type,
    }

    fit_config = {
        'output_dir': args.model_dir,
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'eval_strategy': "steps" if args.eval_steps else "epoch",
        'eval_steps': args.eval_steps,
        'logging_strategy': "steps" if args.eval_steps else "epoch",
        'save_strategy': args.save_strategy,
        'per_device_train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'load_best_model_at_end': load_best_model_at_end,
        'metric_for_best_model': 'eval_spearman_cosine',
        'greater_is_better': True,
        'save_total_limit': 2,
        'lr_scheduler_type': args.lr_scheduler_type,
        'weight_decay': args.weight_decay,
        'dataloader_drop_last': True,
    }

    # Setting the float arguments based on the hardware. If the GPU support it bf16 will be used.
    float_args = model_utils.get_float_args()
    fit_config.update(float_args)

    # Initialize run_settings at once
    run_settings = {"model": "Cross-SBERT", "manual": "1", "no_fit": args.no_fit}
    run_settings.update(datasets_config)
    run_settings.update(pretrained_config)
    run_settings.update(fit_config)

    col_a = 'SentenceA_unicode' if args.use_unicode_columns else 'SentenceA'
    col_b = 'SentenceB_unicode' if args.use_unicode_columns else 'SentenceB'
    if args.use_unicode_columns:
        print(f"Using unicode columns: {col_a}, {col_b}")

    # Getting the datasets.
    df_mapping = {col_a: 'text1', col_b: 'text2', 'score': 'label'}
    train_datasets, valid_ds, test_datasets = text_utils.get_multiple_datasets(train_filenames, args.validation_filename, test_filenames, args.data_dir, args.train_dir, df_mapping, random_state=args.seed)
    print('Datasets loaded')

    # Load pretrained CrossEncoder model
    model = CrossEncoder(model_name_or_path, num_labels=1)
    print(f'Model loaded: {model_name_or_path}')

    log_results_file = os.path.join(args.results_dir, args.results_filename)
    Path(log_results_file).parent.mkdir(parents=True, exist_ok=True)
    _stem = Path(args.results_filename).stem
    _suffix = Path(args.results_filename).suffix
    _per_eval_suffix = '_steps' if args.eval_steps else '_epochs'
    per_epoch_log_results_file = os.path.join(args.results_dir, f"{_stem}{_per_eval_suffix}{_suffix}")
    train_loss = model_utils.get_cross_loss_function(model, args.loss_type)

    if args.no_fit:
        print("No fitting will be done as per --no_fit flag.")
        datasets_config['fit_stage'] = 'none'
        evaluate_and_log(model, train_datasets[0], valid_ds, test_datasets, run_settings, datasets_config, log_results_file)
        finalize(args)
        return

    # # Prepare Evaluator.
    # valid_ds = [
    #     InputExample(
    #         texts=[row["text1"], row["text2"]],
    #         label=float(row["label"])
    #     )
    #     for _, row in valid_ds.iterrows()
    # ]

    # Getting the datasets.
    df_mapping = {col_a: 'text1', col_b: 'text2', 'score': 'label'}
    train_datasets, valid_ds, test_datasets = text_utils.get_multiple_datasets(train_filenames, args.validation_filename, test_filenames, args.data_dir, args.train_dir, df_mapping=df_mapping, eval_df_mapping=None, random_state=args.seed)
    print('Datasets loaded')

    # To Do : need to check this eval method.
    # evaluator = evaluation.CESpearmanEvaluator.from_input_examples(
    #     valid_ds,
    #     name="val-spearman."
    # )
    evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=list(zip(valid_ds["text1"], valid_ds["text2"])),
        scores=valid_ds["label"],
        name="sts_dev",
    )

    for i, train_filename in enumerate(train_filenames):
        interation = '' if i == 0 else str(i)

        train_ds = train_datasets[i]

        # train_ds = [
        #     InputExample(
        #         texts=[row["text1"], row["text2"]],
        #         label=float(row["label"])
        #     )
        #     for _, row in train_ds.iterrows()
        # ]

        fit_config['run_name'] = f"Cross-Encoder-fit{interation}"
        fit_config['metric_for_best_model'] = 'eval_sts_dev_spearman'

        run_settings.update(fit_config)
        print(f"CrossEncoder training on file: {train_filename} with {len(train_ds)} samples.")

        cross_encoder_training_arguments = CrossEncoderTrainingArguments(**fit_config)

        callbacks = []
        if not args.no_per_epoch_eval:
            per_epoch_datasets_config = {
                **datasets_config,
                f'train{interation}_filename': train_filename,
                'validation_filename': args.validation_filename,
                'fit_stage': interation,
            }
            _use_steps = bool(args.eval_steps)
            callbacks.append(_PerEvalCallback(
                model=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_datasets=test_datasets,
                run_settings=run_settings,
                datasets_config=per_epoch_datasets_config,
                log_file=per_epoch_log_results_file,
                use_steps=_use_steps,
            ))
            _granularity = 'step' if _use_steps else 'epoch'
            print(f"Per-{_granularity} evaluation logging ENABLED. Results will be saved at {per_epoch_log_results_file}.")

        trainer = CrossEncoderTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            loss=train_loss,
            evaluator=evaluator,
            args=cross_encoder_training_arguments,
            callbacks=callbacks if callbacks else None,
        )
        trainer.train()
        print(f"CrossEncoder training (Train {i}) complete.")

        # Log best epoch/step and best metric value
        best_metric_value = trainer.state.best_metric
        metric_key = fit_config.get('metric_for_best_model', 'eval_sts_dev_spearman')
        best_step, best_epoch = None, None
        for entry in trainer.state.log_history:
            if entry.get(metric_key) == best_metric_value:
                best_step = entry.get('step')
                best_epoch = entry.get('epoch')
                break
        datasets_config['best_metric_value'] = best_metric_value
        if args.eval_steps:
            datasets_config['best_step'] = best_step
        else:
            datasets_config['best_epoch'] = best_epoch

        # --- EVALUATION ---
        if not load_best_model_at_end:
            datasets_config[f'evaluated_model'] = 'last'
            print("Using the LAST model after first fit.")
        else:
            datasets_config[f'evaluated_model'] = 'best'
            print("Using BEST model after first fit.")

        datasets_config[f'train{interation}_filename'] = train_filename
        datasets_config['validation_filename'] = args.validation_filename
        datasets_config['fit_stage'] = interation
        evaluate_and_log(model, train_ds, valid_ds, test_datasets, run_settings, datasets_config, log_results_file)

    if args.final_model_dir:
        model.save_pretrained(args.final_model_dir)
        print(f"Final model saved to: {args.final_model_dir}")

    finalize(args)

def evaluate_and_log(model, train_ds, valid_ds, test_datasets, run_settings, datasets_config, log_results_file):
    train_metrics = model_utils_pretrain.EMPTY_METRICS
    validation_metrics = model_utils.cross_encoder_evaluate(model, valid_ds, batch_size=4)

    tests_metrics = []
    for i, test_ds in enumerate(test_datasets):
        test_metrics = model_utils.cross_encoder_evaluate(model, test_ds, batch_size=4)
        tests_metrics.append(test_metrics)

    eval_result = model_utils_pretrain.create_multi_tests_eval_result(train_metrics, validation_metrics, tests_metrics, prefix="")
    run_settings.update(datasets_config)

    utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result)


def initialize(args=None):

    keys_filepath = 'keys.yaml'

    hf_token = args.hf_token
    if hf_token is None:
        if os.path.exists(keys_filepath):
            keys_config = AppConfig('keys.yaml')
            hf_token = keys_config.get('HF_TOKEN', None)

    if hf_token:
        login(token=hf_token)
    else:
        login()

    model_utils.initialize_wandb()

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

def finalize(args=None):
    if (not args.keep_previous_model_in_dir) and os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
        print('Deleted previous model!')
    else:
        print('Previous model was kept. (if such exists)')

if __name__ == '__main__':
    main()