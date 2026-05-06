
import argparse
import gc
import shutil
import os
import math
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances

from huggingface_hub import login
from datasets import concatenate_datasets
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from transformers import ModernBertConfig

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


class _FixedSentenceTransformerTrainer(SentenceTransformerTrainer):
    """Restores __len__ on the DataLoader after Accelerate's prepare() drops it.

    Accelerate wraps the DataLoader and can strip __len__, causing the HuggingFace Trainer
    to treat the dataloader as length-less (TypeError on len()), which breaks max_steps
    computation and epoch counting.  This subclass patches the type of the returned
    DataLoader to re-add __len__ using the length computed from the training dataset.
    """

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        try:
            len(dl)
            return dl  # __len__ already works; nothing to do
        except TypeError:
            pass

        n = len(self.train_dataset)
        bs = self.args.per_device_train_batch_size
        drop = self.args.dataloader_drop_last
        length = n // bs if drop else math.ceil(n / bs)

        base = type(dl)
        patched = type(f'_{base.__name__}WithLen', (base,), {'__len__': lambda self, _l=length: _l})
        try:
            dl.__class__ = patched
        except TypeError:
            pass  # can't patch (e.g. C-extension class); training will still crash later
        return dl

disable_progress_bars()


class _PerEpochEvalCallback(TrainerCallback):
    """Evaluates the model after each epoch and appends results to a per-epoch CSV."""

    def __init__(self, st_model, train_ds, valid_ds, test_datasets,
                 run_settings, datasets_config, log_file, run_timestamp=None):
        self._st_model = st_model
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._test_datasets = test_datasets
        self._run_settings = dict(run_settings)    # snapshot
        self._datasets_config = dict(datasets_config)  # snapshot
        self._log_file = log_file
        self._run_timestamp = run_timestamp

    def on_evaluate(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        epoch_datasets_config = {
            **self._datasets_config,
            'evaluated_model': f'epoch_{epoch}',
            'epoch': epoch,
        }
        evaluate_and_log(
            self._st_model, self._train_ds, self._valid_ds, self._test_datasets,
            dict(self._run_settings),
            epoch_datasets_config,
            self._log_file,
            run_timestamp=self._run_timestamp,
        )


def main(args_list=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate sentence similarity')
    # ... (parser arguments remain the same) ...
    parser.add_argument('--data_dir', default='./data', help='The directory that stores the data (defaults to ./data)')
    parser.add_argument('--train_dir', default='./data',
                        help='The directory that stores the train data (defaults to ./data)')
    parser.add_argument('--model_dir', default=None, help='The directory that stores the output model (default: ckpts/sts-b/{timestamp})')
    parser.add_argument('--final_model_dir', default=None,
                        help='If set, save the final trained model to this directory after all training phases complete')
    parser.add_argument('--results_dir', default='./results',
                        help='The directory that stores the results (defaults to ./results)')

    parser.add_argument('--train_filenames', nargs='+', default='train_pairs_B_shuffled_600_scored.xlsx llms_pairs_B_shuffled_2500_scored.xlsx',
                        help='The filenames containing the training data')
    #parser.add_argument('--train2_filename', default=None, help='Optional second training file')

    parser.add_argument('--validation_filename', default='validation_pairs_B_shuffled_150_scored.xlsx',
                        help='The filename containing the validation data')

    parser.add_argument('--test_filenames', nargs='+', default='test_pairs_B_shuffled_250_scored.xlsx test_pairs_B_shuffled_no_positives_scored.xlsx',
                        help='The filenames containing the test data')
    #parser.add_argument('--test2_filename', default=None, help='Optional second test file')

    parser.add_argument('--hf_base_model', default='Intellexus/mbert-tibetan-continual-wylie-final',
                        help='The Hugging Face base model identifier name')
    parser.add_argument('--hf_token', help='The Hugging Face access token')

    parser.add_argument('--results_filename', default='sbert_results.csv', help='The filename to save the results')
    parser.add_argument('--keep_previous_model_in_dir', action='store_true',
                        help='When marked the previous model with the check points will not be deleted')
    parser.add_argument('--seed', type=int, default=42, help='the random seed (defaults to 0)')

    parser.add_argument('--no_fit', default=False, action='store_true', help='Do not do any fitting of the model')
    parser.add_argument('--no_eval', default=False, action='store_true', help='Skip loading validation/test sets and all evaluation steps')
    parser.add_argument('--no_per_epoch_eval', default=False, action='store_true', help='Disable per-epoch evaluation logging to the _epochs CSV')
    parser.add_argument('--use_unicode_columns', default=False, action='store_true',
                        help='Use SentenceA_unicode/SentenceB_unicode columns instead of SentenceA/SentenceB')
    parser.add_argument('--merge_train_files', default=False, action='store_true',
                        help='Merge all train files into a single shuffled dataset and train once instead of sequentially')

    parser.add_argument('--pool_of_pairs_filename',
                        help='The given file will be added a cosine column with the similarity score')
    parser.add_argument('--pool_of_pairs_cosine_filename',
                        help='If pool_of_pairs_filename was given the result will output to this file')

    pretrained_config_group = parser.add_argument_group('Additional for pretrained_config',
                                                        'Additional for pretrained_config')
    pretrained_config_group.add_argument('--pooling_strategy', type=str, default='weightedmean', choices=['cls', 'mean', 'lasttoken', 'max', 'mean_sqrt_len_tokens', 'weightedmean'],
                                         help='pooling_strategy (defaults to mean)')
    pretrained_config_group.add_argument('--max_seq_length', type=int, default=512, help='max_seq_length (defaults to 512)')
    # pretrained_config_group.add_argument('--train_mode', type=bool, default=False,
    #                                      help='train_mode (defaults to False)')
    # pretrained_config_group.add_argument('--local_files_only', type=bool, default=False,
    #                                      help='If True will load the files from local path')

    fit_config_group = parser.add_argument_group('Additional for fit_config', 'Additional for fit_config')
    fit_config_group.add_argument('--batch_size', type=int, default=32, help='batch_size (defaults to 32)')
    fit_config_group.add_argument('--epochs', type=int, default=7, help='epochs (defaults to 7)')
    fit_config_group.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate (defaults to 2e-5)')
    fit_config_group.add_argument('--save_strategy', type=str, default="no", help='save_strategy [no, steps, epoch, best] (defaults to no)')
    fit_config_group.add_argument('--warmup_steps', type=int, default=128, help='warmup_steps (defaults to 128)')
    fit_config_group.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay (defaults to 0.1)')
    fit_config_group.add_argument('--gradient_accumulation_steps', type=int, default=16, help='gradient_accumulation_steps (defaults to 16)')
    fit_config_group.add_argument('--loss_type', type=str, default='cosent', choices=['cosine', 'cosent', 'angleloss', 'mnrl', 'ct'], help='Loss function to use: cosine (CosineSimilarityLoss), cosent (CoSENTLoss), mnrl (MultipleNegativesRankingLoss), ct (ContrastiveTensionLossInBatchNegatives)')
    fit_config_group.add_argument('--loss_scale', type=float, default=20.0, help='Scale factor for CoSENTLoss / AnglELoss (default: 20.0)')
    fit_config_group.add_argument('--lr_scheduler_type', type=str, default='reduce_lr_on_plateau', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'cosine_warmup_with_min_lr', 'warmup_stable_decay'], help='The scheduler type to use')
    fit_config_group.add_argument('--wrong_pairs_log_file', type=str, default=None, help='If provided, pairs with wrong evaluation (|cosine - label| > 0.3) are appended to this CSV file with a run timestamp.')

    args = parser.parse_args(args_list)

    # Initialize env.
    initialize(args)

    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.model_dir is None:
        args.model_dir = f'ckpts/sts-b/{run_timestamp}'

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
    }

    for i, test_filename in enumerate(test_filenames):
        datasets_config[f'test{i}_filename'] = test_filename


    pretrained_config = {
        'model_name_or_path': model_name_or_path,
        'pooling_strategy': args.pooling_strategy,
        "loss_type": args.loss_type,
        "loss_scale": args.loss_scale,
        "use_unicode_columns": args.use_unicode_columns
    }


    _save_strategy = args.save_strategy if not args.no_eval else 'epoch'
    fit_config = {
        'dataloader_num_workers': 1,
        'output_dir': args.model_dir,
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'eval_strategy': "no" if args.no_eval else "epoch",
        'logging_strategy': "epoch",
        'save_strategy': _save_strategy,
        'per_device_train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'load_best_model_at_end': load_best_model_at_end and not args.no_eval,
        'metric_for_best_model': 'eval_spearman_cosine',
        'greater_is_better': True,
        'save_total_limit': 2,
        'lr_scheduler_type': 'linear' if args.no_eval else args.lr_scheduler_type,
        'weight_decay': args.weight_decay,
        **({'lr_scheduler_kwargs': {'min_lr_rate': 0.1}} if (not args.no_eval and args.lr_scheduler_type in ('cosine_with_min_lr', 'cosine_warmup_with_min_lr')) else {}),
        'dataloader_drop_last': True,
        'seed': args.seed,
    }

    # Setting the float arguments based on the hardware. If the GPU support it bf16 will be used.
    float_args = model_utils.get_float_args()
    fit_config.update(float_args)

    # Initialize run_settings at once
    run_settings = {"model": "SBERT", "manual": "1", "no_fit": args.no_fit}
    run_settings.update(datasets_config)
    run_settings.update(pretrained_config)
    run_settings.update(fit_config)

    # Getting the datasets.
    # MNRL uses (anchor, positive) pairs without labels; other losses need score/label.
    eval_df_mapping, train_df_mapping = get_mappings(args)
    if args.no_eval:
        train_datasets, valid_ds, test_datasets = text_utils.get_multiple_datasets(train_filenames, None, [], args.data_dir, args.train_dir, train_df_mapping, eval_df_mapping, random_state=args.seed)
    else:
        train_datasets, valid_ds, test_datasets = text_utils.get_multiple_datasets(train_filenames, args.validation_filename, test_filenames, args.data_dir, args.train_dir, train_df_mapping, eval_df_mapping, random_state=args.seed)
    print('Datasets loaded')

    # load pretrained model and set the pooling.
    transformer = models.Transformer(model_name_or_path, max_seq_length=args.max_seq_length)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=args.pooling_strategy)
    model = SentenceTransformer(modules=[transformer, pooling], model_kwargs={"reference_compile": False}) # , device=gpu_device
    print(f'Model loaded: {model_name_or_path}')

    log_results_file = os.path.join(args.results_dir, args.results_filename)
    Path(log_results_file).parent.mkdir(parents=True, exist_ok=True)
    _stem = Path(args.results_filename).stem
    _suffix = Path(args.results_filename).suffix
    per_epoch_log_results_file = os.path.join(args.results_dir, f"{_stem}_epochs{_suffix}")
    train_loss = model_utils.get_loss_function(model, args.loss_type, scale=args.loss_scale)

    if args.no_fit:
        print("No fitting will be done as per --no_fit flag.")
        datasets_config['fit_stage'] = 'none'
        datasets_config['train_size'] = len(train_datasets[0]) if train_datasets else 0
        evaluate_and_log(model, train_datasets[0], valid_ds, test_datasets, run_settings, datasets_config, log_results_file, wrong_pairs_log_file=args.wrong_pairs_log_file, run_timestamp=run_timestamp)
        finalize(args)
        return

    # Prepare Evaluator.
    if args.no_eval:
        evaluator = None
    else:
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            valid_ds['text1'],
            valid_ds['text2'],
            [float(label) for label in valid_ds['label']],
            name='',
            main_similarity='cosine'
        )

    if args.merge_train_files and len(train_datasets) > 1:
        merged_label = " + ".join(train_filenames)
        merged_ds = concatenate_datasets(train_datasets).shuffle(seed=args.seed)
        train_datasets = [merged_ds]
        train_filenames = [merged_label]
        print(f"Merged train files into one shuffled dataset: {len(merged_ds)} samples.")

    for i, train_filename in enumerate(train_filenames):
        interation = '' if i == 0 else str(i)
        train_ds = train_datasets[i]
        n = len(train_ds)

        # When n < batch_size, drop_last=True would produce 0 batches and len(dataloader)=0.
        # 0 is falsy, so the Trainer treats it as "no length" and errors.  Disable drop_last
        # for this run only so there is at least 1 batch; restore it afterwards.
        _restored_drop_last = False
        if n < args.batch_size and fit_config.get('dataloader_drop_last', False):
            fit_config['dataloader_drop_last'] = False
            _restored_drop_last = True

        fit_config['run_name'] = f"SBERT-{args.loss_type}-fit{interation}"
        run_settings.update(fit_config)
        print(f"SentenceTransformer training on file: {train_filename} with {n} samples.")

        sentence_transformer_training_arguments = SentenceTransformerTrainingArguments(**fit_config)

        # === PATCH PATCH ===
        # Manually add the missing attribute to prevent the crash
        if not hasattr(sentence_transformer_training_arguments, "save_safetensors"):
            sentence_transformer_training_arguments.save_safetensors = False
        # ======================

        callbacks = []
        if not args.no_eval and not args.no_per_epoch_eval:
            per_epoch_datasets_config = {
                **datasets_config,
                f'train{interation}_filename': train_filename,
                f'train{interation}_size': n,
                'fit_stage': interation,
                'validation_filename': args.validation_filename,
            }
            callbacks.append(_PerEpochEvalCallback(
                st_model=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_datasets=test_datasets,
                run_settings=run_settings,
                datasets_config=per_epoch_datasets_config,
                log_file=per_epoch_log_results_file,
                run_timestamp=run_timestamp,
            ))

        trainer = _FixedSentenceTransformerTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            loss=train_loss,
            evaluator=evaluator,
            args=sentence_transformer_training_arguments,
            callbacks=callbacks or None,
        )
        trainer.train()
        print(f"SentenceTransformer training (Train {i}) complete.")
        if args.save_strategy == "no":
            model.save(args.model_dir)
            print(f"Model saved to {args.model_dir}")

        if _restored_drop_last:
            fit_config['dataloader_drop_last'] = True

        datasets_config.update(get_best_checkpoint_info(args.model_dir))

        # --- EVALUATION ---
        if not load_best_model_at_end:
            datasets_config[f'evaluated_model'] = 'last'
            print("Using the LAST model after first fit.")
        else:
            datasets_config[f'evaluated_model'] = 'best'
            print("Using BEST model after first fit.")

        datasets_config[f'train{interation}_filename'] = train_filename
        datasets_config[f'train{interation}_size'] = n
        datasets_config[f'validation_filename'] = args.validation_filename
        datasets_config[f'fit_stage'] = interation
        if not args.no_eval:
            datasets_config[f'validation_filename'] = args.validation_filename
            evaluate_and_log(model, train_ds, valid_ds, test_datasets, run_settings, datasets_config, log_results_file, wrong_pairs_log_file=args.wrong_pairs_log_file, run_timestamp=run_timestamp)

    if args.final_model_dir:
        model.save(args.final_model_dir)
        print(f"Final model saved to: {args.final_model_dir}")

    del model, train_loss, evaluator
    torch.cuda.empty_cache()
    gc.collect()

    finalize(args)


def get_best_checkpoint_info(model_dir: str) -> dict:
    """
    Read {model_dir}/eval/similarity_evaluation_results.csv and return the
    epoch, steps and cosine_spearman of the row with the highest cosine_spearman.
    Returns an empty dict if the file is missing or unreadable.
    """
    eval_csv = os.path.join(model_dir, "eval", "similarity_evaluation_results.csv")
    try:
        df = pd.read_csv(eval_csv)
        best_row = df.loc[df["cosine_spearman"].idxmax()]
        return {
            "best_epoch": best_row["epoch"],
            "best_steps": int(best_row["steps"]),
            "best_cosine_spearman": round(float(best_row["cosine_spearman"]), 4),
        }
    except Exception as e:
        print(f"Warning: could not read best checkpoint info from {eval_csv}: {e}")
        return {}


def get_mappings(args: Namespace) -> tuple[dict[str, str], dict[str, str]]:
    col_a = 'SentenceA_unicode' if args.use_unicode_columns else 'SentenceA'
    col_b = 'SentenceB_unicode' if args.use_unicode_columns else 'SentenceB'
    eval_df_mapping = {col_a: 'text1', col_b: 'text2', 'score': 'label'}
    if args.loss_type in ('mnrl', 'ct'):
        train_df_mapping = {col_a: 'text1', col_b: 'text2'}
    else:
        train_df_mapping = eval_df_mapping
    return eval_df_mapping, train_df_mapping


def evaluate_and_log(model, train_ds, valid_ds, test_datasets, run_settings, datasets_config, log_results_file, batch_size=4, wrong_pairs_log_file=None, run_timestamp=None):
    train_metrics = model_utils_pretrain.EMPTY_METRICS
    validation_metrics = model_utils_pretrain.bi_encoder_evaluate(model, valid_ds, batch_size=4)

    tests_metrics = []
    for i, test_ds in enumerate(test_datasets):
        test_metrics = model_utils_pretrain.bi_encoder_evaluate(model, test_ds, batch_size=4)
        tests_metrics.append(test_metrics)

    eval_result = model_utils_pretrain.create_multi_tests_eval_result(train_metrics, validation_metrics, tests_metrics, prefix="")

    if 'label' in train_ds.column_names:
        train_scores = np.array([float(s) for s in train_ds['label']])
        eval_result['train_score_mean'] = round(float(train_scores.mean()), 4)
        eval_result['train_score_std'] = round(float(train_scores.std()), 4)

    run_settings.update(datasets_config)

    utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result)

    if wrong_pairs_log_file:
        _log_wrong_pairs(model, valid_ds, dataset_name='validation', run_timestamp=run_timestamp, log_file=wrong_pairs_log_file, batch_size=batch_size)
        for i, test_ds in enumerate(test_datasets):
            _log_wrong_pairs(model, test_ds, dataset_name=f'test{i}', run_timestamp=run_timestamp, log_file=wrong_pairs_log_file, batch_size=batch_size)


def _log_wrong_pairs(model, ds, dataset_name, run_timestamp, log_file, batch_size=4, error_threshold=0.3):
    """Encode all pairs in ds, find those where |cosine - label| > error_threshold, and append them to log_file."""
    from boltons.iterutils import chunked_iter

    text1 = ds['text1']
    text2 = ds['text2']
    labels = [float(l) for l in ds['label']]
    ids = ds['ID'] if 'ID' in ds.column_names else [None] * len(text1)

    emb1_chunks, emb2_chunks = [], []
    for chunk in chunked_iter(range(len(text1)), batch_size):
        emb1_chunks.append(model.encode([text1[i] for i in chunk]))
        emb2_chunks.append(model.encode([text2[i] for i in chunk]))

    emb1 = np.concatenate(emb1_chunks, axis=0)
    emb2 = np.concatenate(emb2_chunks, axis=0)
    cosines = 1 - paired_cosine_distances(emb1, emb2)

    rows = []
    for idx in range(len(text1)):
        error = cosines[idx] - labels[idx]
        if abs(error) > error_threshold:
            rows.append({
                'run_timestamp': run_timestamp,
                'dataset': dataset_name,
                'ID': ids[idx],
                'text1': text1[idx],
                'text2': text2[idx],
                'label': labels[idx],
                'predicted_cosine': round(float(cosines[idx]), 4),
                'error': round(float(error), 4),
            })

    if not rows:
        return

    wrong_df = pd.DataFrame(rows)
    write_header = not os.path.exists(log_file)
    wrong_df.to_csv(log_file, mode='a', index=False, header=write_header)
    print(f"Logged {len(rows)} wrong pairs ({dataset_name}) -> {log_file}")


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