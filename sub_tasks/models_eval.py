"""
Evaluate and train models on the four sets (A, B, C, D).

- evaluate_model_on_sets: Evaluate Qwen3-Embedding on 4 sets (no training).
- train_mnrl_and_evaluate: Train mBERT with MultipleNegativesRankingLoss on
  consecutive segment pairs from kangyur_tangyur_cleaned_segments.xlsx, then
  evaluate on the 4 sets.
- evaluate_local_llm_on_4_pairs: Annotate a 4-pair file with a local LLM
  (e.g. Qwen/Qwen3-32B-FP8) via vLLM for BWS similarity evaluation.

Requires:
    transformers>=4.51.0
    sentence-transformers>=2.7.0
    vllm  (for evaluate_local_llm_on_4_pairs)

Usage:
    python -m sub_tasks.models_eval
"""
from common_utils import gpu_utils


def update_env():
    import subprocess
    import sys
    import os

    modules = ["pandas", "transformers", "sentence-transformers", "openpyxl", "datasets", "boltons", "ipython", "weave", "langchain"]

    for module in modules:
        try:
            # We use '-n base' to target the active environment in the container
            # We use '--prune' to remove packages not in the yaml (optional, remove if unsafe)
            print(f'🔄 Updating environment: installing {module}...')
            command = ["pip", "install", f"{module}"]

            # Execute the command and wait for it to finish
            subprocess.check_call(command, stdout=sys.stdout, stderr=sys.stderr)

            print("✅ Environment updated successfully.")

        except subprocess.CalledProcessError:
            print("❌ Failed to update environment")
            sys.exit(1)  # Stop execution if dependencies fail

    print(f'finished updating environment. Current working directory: {os.getcwd()}')

#update_env()

import re
import sys
import os
import shutil
import random
from pathlib import Path
from datetime import datetime

import torch
import gc

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sub_tasks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sensim/

import text_utils
import model_utils
import model_utils_pretrain
import utils
from app_config import AppConfig
import aggregate_sets
import sbert_with_pretrain
from mnrl_training import generate_consecutive_pairs, train_mnrl_and_evaluate

gpu_device = model_utils.get_gpu_device()


def evaluate_and_log(model, train_ds, valid_ds, test_datasets, run_settings, datasets_config, log_results_file):
    train_metrics = model_utils_pretrain.EMPTY_METRICS
    validation_metrics = model_utils_pretrain.bi_encoder_evaluate(model, valid_ds, batch_size=4)

    tests_metrics = []
    for i, test_ds in enumerate(test_datasets):
        test_metrics = model_utils_pretrain.bi_encoder_evaluate(model, test_ds, batch_size=4)
        tests_metrics.append(test_metrics)

    eval_result = model_utils_pretrain.create_multi_tests_eval_result(train_metrics, validation_metrics, tests_metrics, prefix="")
    run_settings.update(datasets_config)

    utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result)



def reranker_evaluate(model_name, data, instruction, batch_size=4, max_length=8192):
    """
    Evaluate a reranker model (e.g. Qwen3-Reranker) using compute_logits.

    Returns a metrics dict matching the shape of bi_encoder_evaluate / cross_encoder_evaluate.
    """

    gpu_utils.clean_memory_with_info()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _device_map = "auto" if num_gpus > 1 else gpu_device

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=_device_map
    ).eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    def format_instruction(inst, query, doc):
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=inst, query=query, doc=doc
        )

    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(inputs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()

    # Score all pairs in batches
    text1_list = list(data['text1'])
    text2_list = list(data['text2'])
    gold_scores = list(data['label'])

    all_pred_scores = []
    for start in range(0, len(text1_list), batch_size):
        end = min(start + batch_size, len(text1_list))
        pairs = [
            format_instruction(instruction, text1_list[i], text2_list[i])
            for i in range(start, end)
        ]
        inputs = process_inputs(pairs)
        batch_scores = compute_logits(inputs)
        all_pred_scores.extend(batch_scores)

    pred_scores = np.array(all_pred_scores)
    pearson_cosine, pearson_pvalue = pearsonr(gold_scores, pred_scores)
    spearman_cosine, spearman_pvalue = spearmanr(gold_scores, pred_scores)
    kendall_cosine, kendall_pvalue = kendalltau(gold_scores, pred_scores)

    metrics = {
        "pearson_cosine": pearson_cosine,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_cosine,
        "spearman_pvalue": spearman_pvalue,
        "kendall_cosine": kendall_cosine,
        "kendall_pvalue": kendall_pvalue,
        "pearson_manhattan": pearson_cosine,
        "spearman_manhattan": spearman_cosine,
        "kendall_manhattan": kendall_cosine,
        "pearson_euclidean": pearson_cosine,
        "spearman_euclidean": spearman_cosine,
        "kendall_euclidean": kendall_cosine,
        "pearson_dot": pearson_cosine,
        "spearman_dot": spearman_cosine,
        "kendall_dot": kendall_cosine,
    }

    # Clean up: remove accelerate dispatch hooks before moving to CPU
    from accelerate.hooks import remove_hook_from_submodules
    remove_hook_from_submodules(model)
    model.cpu()
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def bge_reranker_evaluate(model_name, data, instruction, batch_size=4, max_length=1024):
    """
    Evaluate a BGE-style LLM reranker (e.g. BAAI/bge-reranker-v2-gemma).

    Scores pairs using the logit at the 'Yes' token position of the last output token.
    Returns a metrics dict matching the shape of bi_encoder_evaluate / cross_encoder_evaluate.
    """
    clean_mem()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _device_map = "auto" if num_gpus > 1 else gpu_device

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=_device_map
    ).eval()

    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    sep_inputs = tokenizer("\n", return_tensors=None, add_special_tokens=False)['input_ids']
    prompt_inputs = tokenizer(instruction, return_tensors=None, add_special_tokens=False)['input_ids']

    def get_inputs(pairs):
        inputs = []
        for query, passage in pairs:
            query_inputs = tokenizer(
                f'A: {query}', return_tensors=None, add_special_tokens=False,
                max_length=max_length * 3 // 4, truncation=True
            )
            passage_inputs = tokenizer(
                f'B: {passage}', return_tensors=None, add_special_tokens=False,
                max_length=max_length, truncation=True
            )
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return tokenizer.pad(
            inputs, padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8, return_tensors='pt',
        )

    # Score all pairs in batches
    text1_list = list(data['text1'])
    text2_list = list(data['text2'])
    gold_scores = list(data['label'])

    all_pred_scores = []
    for start in range(0, len(text1_list), batch_size):
        end = min(start + batch_size, len(text1_list))
        pairs = [(text1_list[i], text2_list[i]) for i in range(start, end)]
        with torch.no_grad():
            inputs = get_inputs(pairs)
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)
            scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1).float().tolist()
        all_pred_scores.extend(scores)

    pred_scores = np.array(all_pred_scores)
    pearson_cosine, pearson_pvalue = pearsonr(gold_scores, pred_scores)
    spearman_cosine, spearman_pvalue = spearmanr(gold_scores, pred_scores)
    kendall_cosine, kendall_pvalue = kendalltau(gold_scores, pred_scores)

    metrics = {
        "pearson_cosine": pearson_cosine,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_cosine,
        "spearman_pvalue": spearman_pvalue,
        "kendall_cosine": kendall_cosine,
        "kendall_pvalue": kendall_pvalue,
        "pearson_manhattan": pearson_cosine,
        "spearman_manhattan": spearman_cosine,
        "kendall_manhattan": kendall_cosine,
        "pearson_euclidean": pearson_cosine,
        "spearman_euclidean": spearman_cosine,
        "kendall_euclidean": kendall_cosine,
        "pearson_dot": pearson_cosine,
        "spearman_dot": spearman_cosine,
        "kendall_dot": kendall_cosine,
    }

    # Clean up: remove accelerate dispatch hooks before moving to CPU
    from accelerate.hooks import remove_hook_from_submodules
    remove_hook_from_submodules(model)
    model.cpu()
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def _clear_distributed_env():
    """Remove torchrun/DDP env vars so the subprocess gets clean GPU access."""
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "LOCAL_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
                "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
                "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_ERROR_FILE"):
        os.environ.pop(key, None)


def _bi_encoder_predict_worker(model_name, data, batch_size, result_file):
    """Worker that runs in a subprocess. Writes scores to a temp .npy file."""
    _clear_distributed_env()
    from sklearn.metrics.pairwise import paired_cosine_distances

    _gpu = model_utils.get_gpu_device()
    model = SentenceTransformer(model_name, device=None, model_kwargs={"dtype": torch.bfloat16})

    text1 = list(data['text1'])
    text2 = list(data['text2'])

    embeddings1 = model.encode(text1, batch_size=batch_size, show_progress_bar=True)
    embeddings2 = model.encode(text2, batch_size=batch_size, show_progress_bar=True)

    embeddings1 = np.nan_to_num(embeddings1, nan=0.0, posinf=1.0, neginf=-1.0)
    embeddings2 = np.nan_to_num(embeddings2, nan=0.0, posinf=1.0, neginf=-1.0)

    scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
    if result_file:
        np.save(result_file, np.array(scores))
    else:
        return scores


def bi_encoder_predict(model_name, data, batch_size=4):
    """Return raw cosine similarity scores from a bi-encoder model.

    Runs in a subprocess so GPU memory is fully freed on exit.
    """
    import tempfile
    import multiprocessing as mp

    gpu_utils.clean_memory_with_info()
    scores = _bi_encoder_predict_worker(model_name, data, batch_size, result_file=None)
    return np.array(scores)

    # # warm up GPU and cache before spawning subprocess
    #
    # ctx = mp.get_context("spawn")
    #
    # with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
    #     result_file = f.name
    #
    # try:
    #     p = ctx.Process(target=_bi_encoder_predict_worker,
    #                     args=(model_name, data, batch_size, result_file))
    #     p.start()
    #     p.join()
    #     if p.exitcode != 0:
    #         raise RuntimeError(f"bi_encoder_predict subprocess failed with exit code {p.exitcode}")
    #     scores = np.load(result_file)
    # finally:
    #     import os as _os
    #     if _os.path.exists(result_file):
    #         _os.unlink(result_file)
    #
    # return scores


def _reranker_predict_worker(model_name, data, instruction, batch_size, max_length, result_file):
    """Worker that runs in a subprocess. Writes scores to a temp .npy file."""
    _clear_distributed_env()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _gpu = model_utils.get_gpu_device()
    device_map = "auto" if num_gpus > 1 else _gpu

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device_map
    ).eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    def format_instruction(inst, query, doc):
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=inst, query=query, doc=doc
        )

    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(inputs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()

    text1_list = list(data['text1'])
    text2_list = list(data['text2'])

    all_pred_scores = []
    for start in range(0, len(text1_list), batch_size):
        end = min(start + batch_size, len(text1_list))
        pairs = [
            format_instruction(instruction, text1_list[i], text2_list[i])
            for i in range(start, end)
        ]
        inputs = process_inputs(pairs)
        batch_scores = compute_logits(inputs)
        all_pred_scores.extend(batch_scores)

    if result_file:
        np.save(result_file, np.array(all_pred_scores))
    return all_pred_scores


def reranker_predict(model_name, data, instruction, batch_size=4, max_length=8192):
    """Return raw reranker scores from a Qwen-style reranker model.

    Runs in a subprocess so GPU memory is fully freed on exit.
    """

    gpu_utils.clean_memory_with_info()
    scores = _reranker_predict_worker(model_name, data, instruction, batch_size, max_length, result_file=None)
    return np.array(scores)

    # import tempfile
    # import multiprocessing as mp
    #
    # clean_mem()
    # ctx = mp.get_context("spawn")
    #
    # with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
    #     result_file = f.name
    #
    # try:
    #     p = ctx.Process(target=_reranker_predict_worker,
    #                     args=(model_name, data, instruction, batch_size, max_length, result_file))
    #     p.start()
    #     p.join()
    #     if p.exitcode != 0:
    #         raise RuntimeError(f"reranker_predict subprocess failed with exit code {p.exitcode}")
    #     scores = np.load(result_file)
    # finally:
    #     import os as _os
    #     if _os.path.exists(result_file):
    #         _os.unlink(result_file)
    #
    # return scores


def scores_to_metrics(gold_scores, pred_scores):
    """Compute the standard metrics dict from gold and predicted scores."""
    pearson_cosine, pearson_pvalue = pearsonr(gold_scores, pred_scores)
    spearman_cosine, spearman_pvalue = spearmanr(gold_scores, pred_scores)
    kendall_cosine, kendall_pvalue = kendalltau(gold_scores, pred_scores)
    return {
        "pearson_cosine": pearson_cosine,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_cosine,
        "spearman_pvalue": spearman_pvalue,
        "kendall_cosine": kendall_cosine,
        "kendall_pvalue": kendall_pvalue,
        "pearson_manhattan": pearson_cosine,
        "spearman_manhattan": spearman_cosine,
        "kendall_manhattan": kendall_cosine,
        "pearson_euclidean": pearson_cosine,
        "spearman_euclidean": spearman_cosine,
        "kendall_euclidean": kendall_cosine,
        "pearson_dot": pearson_cosine,
        "spearman_dot": spearman_cosine,
        "kendall_dot": kendall_cosine,
    }


def evaluate_model_on_sets(run_time_identifier=None, use_unicode_columns=False):
    """
    Evaluate one or more embedding models on the four sets (A, B, C, D)
    without any fine-tuning, similar to train_and_score_model_with_mined_pairs
    in model_tasks.py.

    Args:
        use_unicode_columns: If True, read SentenceA_unicode/SentenceB_unicode
                             instead of SentenceA/SentenceB.
    """

    bi_encoder_model_names = [
        "BAAI/bge-multilingual-gemma2",
        "Kingsoft-LLM/QZhou-Embedding",
        "sentence-transformers/sentence-t5-xxl",
        "Qwen/Qwen3-Embedding-8B",
        "google/embeddinggemma-300m",
        "Octen/Octen-Embedding-8B",
        "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "BAAI/bge-m3",
    ]

    cross_encoder_model_names = [
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ]

    reranker_models = [
        {
            "name": "Qwen/Qwen3-Reranker-8B",
            "instruction": "Given a web search query, retrieve relevant passages that answer the query",
            "eval_fn": "qwen",
        },
        {
            "name": "BAAI/bge-reranker-v2-gemma",
            "instruction": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.",
            "eval_fn": "bge",
        },
        {
           "name": "Qwen/Qwen3-Reranker-0.6B",
           "instruction": "Given two Tibetan text segments, judge whether they are semantically similar",
           "eval_fn": "qwen",
        },
    ]

    # Combined model evaluation: blend scores from multiple models.
    # Scores are min-max normalized then combined as weighted sum.
    # Each entry in "models" has a type ("bi_encoder" or "reranker") and optional instruction.
    # "weights" is a list of weight tuples (one weight per model, must sum to 1).
    combined_models = [
        {
            "models": [
                {"name": "BAAI/bge-multilingual-gemma2", "type": "bi_encoder"},
                {"name": "google/embeddinggemma-300m", "type": "bi_encoder"},
                #{"name": "BAAI/bge-m3", "type": "bi_encoder"},
                {"name": "Qwen/Qwen3-Reranker-8B", "type": "reranker",
                 "instruction": "Given a web search query, retrieve relevant passages that answer the query"},
                #{"name": "BAAI/bge-reranker-v2-gemma", "type": "reranker",
                # "instruction": "Given a passage A and a passage B, determine whether the passages semantically similar by providing a prediction of either 'Yes' or 'No'."},
            ],
            "weights": [
                (0.33, 0.33, 0.34),
                (0.3, 0.3, 0.4)
            ]
            # "weights": [
            #     # All 4 models
            #     (0.25, 0.25, 0.25, 0.25),
            #     (0.4, 0.2, 0.1, 0.3),
            #     (0.3, 0.2, 0.1, 0.4),
            #     (0.3, 0.1, 0.2, 0.4),
            #     (0.2, 0.2, 0.2, 0.4),
            #     # Without bge-m3
            #     (0.4, 0.3, 0.0, 0.3),
            #     (0.3, 0.3, 0.0, 0.4),
            #     (0.5, 0.25, 0.0, 0.25),
            #     # Without embeddinggemma
            #     (0.4, 0.0, 0.3, 0.3),
            #     (0.3, 0.0, 0.3, 0.4),
            #     (0.5, 0.0, 0.25, 0.25),
            #     # Without reranker
            #     (0.4, 0.3, 0.3, 0.0),
            #     (0.5, 0.25, 0.25, 0.0),
            #     (0.34, 0.33, 0.33, 0.0),
            #     # Without bge-multilingual-gemma2
            #     (0.0, 0.3, 0.3, 0.4),
            #     (0.0, 0.25, 0.25, 0.5),
            #     (0.0, 0.34, 0.33, 0.33),
            #     # Pairs only
            #     (0.5, 0.0, 0.0, 0.5),
            #     (0.0, 0.5, 0.0, 0.5),
            #     (0.0, 0.0, 0.5, 0.5),
            #     (0.5, 0.5, 0.0, 0.0),
            #     (0.5, 0.0, 0.5, 0.0),
            #     (0.0, 0.5, 0.5, 0.0),
            # ],
        },
    ]

    bi_encoder_model_names = []
    cross_encoder_model_names = []
    reranker_models = []
    #combined_models = []

    app_config = AppConfig()
    sensim_base_dir = app_config.get('sensim_base_dir', str(Path(__file__).resolve().parent.parent))
    data_dir = os.path.join(sensim_base_dir, "data", "NewDataA-D")
    results_filepath = os.path.join(sensim_base_dir, "results", f"eval_multi_models_results_{run_time_identifier}.csv")

    test_filename = "all_gold_pairs_1000_scored.xlsx"

    col_a = 'SentenceA_unicode' if use_unicode_columns else 'SentenceA'
    col_b = 'SentenceB_unicode' if use_unicode_columns else 'SentenceB'
    df_mapping = {col_a: 'text1', col_b: 'text2', 'score': 'label'}


    # Load dataset once (shared by all models)
    test_path = os.path.join(data_dir, test_filename)
    test_ds = utils.get_dataset(test_path, df_mapping, random_state=42)

    # Evaluate bi-encoder models
    for model_name in bi_encoder_model_names:

        clean_mem()

        print(f"\n{'#' * 60}")
        print(f"Loading bi-encoder: {model_name}")
        print(f"{'#' * 60}")
        model = SentenceTransformer(model_name, device=gpu_device)
        print(f"Model loaded: {model_name}")

        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name} on: test={test_filename}")
        print(f"{'=' * 60}")

        test_metrics = model_utils_pretrain.bi_encoder_evaluate(model, test_ds, batch_size=4)

        train_metrics = model_utils_pretrain.EMPTY_METRICS
        validation_metrics = model_utils_pretrain.EMPTY_METRICS
        eval_result = model_utils_pretrain.create_multi_tests_eval_result(
            train_metrics, validation_metrics, [test_metrics], prefix=""
        )

        run_settings = _make_run_settings(
            test_filename, model_name, "bi-encoder", pooling_strategy="weightedmean"
        )

        utils.log_evaluation_results(
            log_file=results_filepath, settings=run_settings, results=eval_result
        )

        print(f"  Test Spearman: {test_metrics['spearman_cosine']:.4f}")
        print(f"Done. Results saved to: {results_filepath}")

        del model

    # Evaluate cross-encoder models
    for model_name in cross_encoder_model_names:

        clean_mem()

        print(f"\n{'#' * 60}")
        print(f"Loading cross-encoder: {model_name}")
        print(f"{'#' * 60}")
        model = CrossEncoder(model_name, device=gpu_device)
        print(f"Model loaded: {model_name}")

        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name} on: test={test_filename}")
        print(f"{'=' * 60}")

        test_metrics = model_utils_pretrain.cross_encoder_evaluate(model, test_ds, batch_size=4)

        train_metrics = model_utils_pretrain.EMPTY_METRICS
        validation_metrics = model_utils_pretrain.EMPTY_METRICS
        eval_result = model_utils_pretrain.create_multi_tests_eval_result(
            train_metrics, validation_metrics, [test_metrics], prefix=""
        )

        run_settings = _make_run_settings(test_filename, model_name, "cross-encoder")

        utils.log_evaluation_results(
            log_file=results_filepath, settings=run_settings, results=eval_result
        )

        print(f"  Test Spearman: {test_metrics['spearman_cosine']:.4f}")
        print(f"Done. Results saved to: {results_filepath}")

        del model

    # Evaluate reranker models
    for reranker in reranker_models:

        clean_mem()

        model_name = reranker["name"]
        instruction = reranker["instruction"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\n{'#' * 60}")
        print(f"Loading reranker: {model_name}")
        print(f"Instruction: {instruction}")
        print(f"{'#' * 60}")

        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name} on: test={test_filename}")
        print(f"{'=' * 60}")

        eval_fn = reranker["eval_fn"]
        if eval_fn == "bge":
            test_metrics = bge_reranker_evaluate(model_name, test_ds, instruction, batch_size=4)
        else:
            test_metrics = reranker_evaluate(model_name, test_ds, instruction, batch_size=4)

        train_metrics = model_utils_pretrain.EMPTY_METRICS
        validation_metrics = model_utils_pretrain.EMPTY_METRICS
        eval_result = model_utils_pretrain.create_multi_tests_eval_result(
            train_metrics, validation_metrics, [test_metrics], prefix=""
        )

        run_settings = _make_run_settings(
            test_filename, model_name, "reranker", instruction=instruction
        )

        utils.log_evaluation_results(
            log_file=results_filepath, settings=run_settings, results=eval_result
        )

        print(f"  Test Spearman: {test_metrics['spearman_cosine']:.4f}")
        print(f"Done. Results saved to: {results_filepath}")

    # Evaluate combined models
    gold_scores = list(test_ds['label'])

    def normalize(scores):
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-9:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    for combo in combined_models:
        clean_mem()

        models_cfg = combo["models"]
        weight_sets = combo["weights"]

        model_names = [m["name"] for m in models_cfg]
        combo_label = " + ".join(m["name"].rsplit("/", 1)[-1] for m in models_cfg)

        print(f"\n{'#' * 60}")
        print(f"Combined evaluation: {combo_label}")
        print(f"{'#' * 60}")

        # Get raw scores from each model (one at a time to manage GPU memory)
        all_norm_scores = []
        for m in models_cfg:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            name = m["name"]
            mtype = m["type"]
            print(f"Getting scores from {name} ({mtype})...")

            if mtype == "bi_encoder":
                raw = bi_encoder_predict(name, test_ds, batch_size=4)
            elif mtype == "reranker":
                raw = reranker_predict(name, test_ds, m["instruction"], batch_size=4)
            else:
                raise ValueError(f"Unknown model type: {mtype}")

            all_norm_scores.append(normalize(raw))

        # Evaluate each weight combination
        for ws in weight_sets:
            combined = sum(w * s for w, s in zip(ws, all_norm_scores))
            test_metrics = scores_to_metrics(gold_scores, combined)

            train_metrics = model_utils_pretrain.EMPTY_METRICS
            validation_metrics = model_utils_pretrain.EMPTY_METRICS
            eval_result = model_utils_pretrain.create_multi_tests_eval_result(
                train_metrics, validation_metrics, [test_metrics], prefix=""
            )

            weights_str = ",".join(f"{w:.2f}" for w in ws)
            encoder_type = "combined(" + ",".join(
                f"{m['name'].rsplit('/', 1)[-1]}={w:.2f}" for m, w in zip(models_cfg, ws)
            ) + ")"

            # Per-model weight columns: "weight_<short_name>" = weight value
            weight_cols = {}
            for m, w in zip(models_cfg, ws):
                short = m["name"].rsplit("/", 1)[-1]
                weight_cols[f"weight_{short}"] = f"{w:.2f}"

            run_settings = _make_run_settings(
                test_filename, " + ".join(model_names), encoder_type, **weight_cols
            )

            utils.log_evaluation_results(
                log_file=results_filepath, settings=run_settings, results=eval_result
            )

            print(f"  weights=({weights_str}) -> Spearman: {test_metrics['spearman_cosine']:.4f}")

        print(f"Done. Results saved to: {results_filepath}")


def _make_run_settings(test_filename, model_name_or_path, encoder_type, pooling_strategy="none", **extra):
    """Build the common run_settings dict for evaluate_model_on_sets logging."""
    settings = {
        "model": "SBERT",
        "encoder_type": encoder_type,
        "manual": "1",
        "no_fit": True,
        "model_name_or_path": model_name_or_path,
        "learning_rate": "2e-5",
        "gradient_accumulation_steps": "1",
        "pooling_strategy": pooling_strategy,
        "loss_type": "none",
        "fit_stage": "none",
        "validation_filename": "",
        "test0_filename": test_filename,
        "train_filename": "",
        "train1_filename": "",
    }
    settings.update(extra)
    return settings


def clean_mem():
    # Clear Python garbage
    gc.collect()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 models may produce."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def evaluate_local_llm_on_4_pairs(
    input_file: str,
    output_file: str,
    model_name: str = "Qwen/Qwen3-32B-FP8",
    temperature: float = 0.0,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1,
    trials_if_hallucinated: int = 3,
    seed: int = 42,
    enable_thinking: bool = False,
) -> pd.DataFrame:
    """Annotate a 4-pair file with a local LLM via vLLM.

    Uses the same BWS prompts and parsing logic as llms/llms_eval.py
    but runs entirely locally with no API keys required.

    Args:
        input_file: Path to the 4-pair Excel file (must have pair_*_A/B and id_* columns).
        output_file: Path where the annotated results will be saved (.xlsx).
        model_name: HuggingFace model ID for vLLM (default: Qwen/Qwen3-32B-FP8).
        temperature: Sampling temperature (0 = greedy/deterministic).
        max_tokens: Maximum tokens to generate per response.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        trials_if_hallucinated: Max attempts per row before random fallback.
        seed: Random seed for reproducibility.
        enable_thinking: Whether to enable Qwen3 thinking mode (default: False).

    Returns:
        DataFrame with best_pair, worst_pair, and trials columns added.
    """

    #from vllm import LLM, SamplingParams
    # from llms.prompts_bws import (
    #     SYSTEM_PROMPT, HUMAN_PROMPT_TEMPLATE,
    #     parse_response, row_to_prompt_row, validate_ids,
    # )
    #
    # rng = random.Random(seed)
    #
    # # --- Load and validate input ---
    # input_path = Path(input_file)
    # if not input_path.exists():
    #     raise FileNotFoundError(f"Input file not found: {input_file}")
    #
    # df = pd.read_excel(input_file, engine="openpyxl")
    # rows = df.to_dict(orient="records")
    #
    # required_cols = [
    #     "pair_1_A", "pair_1_B", "pair_2_A", "pair_2_B",
    #     "pair_3_A", "pair_3_B", "pair_4_A", "pair_4_B",
    #     "id_1", "id_2", "id_3", "id_4",
    # ]
    # for col in required_cols:
    #     if col not in df.columns:
    #         raise ValueError(f"Input file is missing required column: {col}")
    #
    # print(f"Loaded {len(rows)} rows from {input_file}")
    #
    # # --- Build chat conversations for all rows ---
    # conversations = []
    # for row in rows:
    #     row_to_prompt_row(row)
    #     human_msg = HUMAN_PROMPT_TEMPLATE.format(
    #         Sentence_1A=row["pair_1_A"], Sentence_1B=row["pair_1_B"],
    #         Sentence_2A=row["pair_2_A"], Sentence_2B=row["pair_2_B"],
    #         Sentence_3A=row["pair_3_A"], Sentence_3B=row["pair_3_B"],
    #         Sentence_4A=row["pair_4_A"], Sentence_4B=row["pair_4_B"],
    #     )
    #     conversations.append([
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": human_msg},
    #     ])
    #
    # # --- Initialize vLLM ---
    # print(f"Loading model: {model_name} (tensor_parallel_size={tensor_parallel_size})")
    # llm = LLM(
    #     model=model_name,
    #     tensor_parallel_size=tensor_parallel_size,
    #     seed=seed,
    #     trust_remote_code=True,
    # )
    #
    # sampling_params = SamplingParams(
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     seed=seed if temperature > 0 else None,
    # )
    #
    # chat_template_kwargs = {}
    # if not enable_thinking:
    #     chat_template_kwargs["enable_thinking"] = False
    #
    # print(f"Model loaded. Processing {len(conversations)} rows...")
    #
    # # --- Initial batch inference ---
    # outputs = llm.chat(
    #     conversations,
    #     sampling_params=sampling_params,
    #     chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
    # )
    #
    # results: list = [None] * len(rows)
    # failed_indices = []
    #
    # for i, (row, output) in enumerate(zip(rows, outputs)):
    #     text = _strip_thinking_tags(output.outputs[0].text)
    #     valid_ids = [row["id_1"], row["id_2"], row["id_3"], row["id_4"]]
    #     most_label, least_label = parse_response(text)
    #     ok, most_id, least_id = validate_ids(most_label, least_label, valid_ids)
    #     if ok:
    #         results[i] = (most_id, least_id, "1", text)
    #     else:
    #         failed_indices.append(i)
    #
    # print(f"  Initial pass: {len(rows) - len(failed_indices)}/{len(rows)} parsed OK")
    #
    # # --- Retry failed rows ---
    # for attempt in range(2, trials_if_hallucinated + 1):
    #     if not failed_indices:
    #         break
    #
    #     retry_convos = [conversations[i] for i in failed_indices]
    #     retry_outputs = llm.chat(
    #         retry_convos,
    #         sampling_params=sampling_params,
    #         chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
    #     )
    #
    #     still_failed = []
    #     for idx, output in zip(failed_indices, retry_outputs):
    #         text = _strip_thinking_tags(output.outputs[0].text)
    #         row = rows[idx]
    #         valid_ids = [row["id_1"], row["id_2"], row["id_3"], row["id_4"]]
    #         most_label, least_label = parse_response(text)
    #         ok, most_id, least_id = validate_ids(most_label, least_label, valid_ids)
    #         if ok:
    #             results[idx] = (most_id, least_id, str(attempt), text)
    #         else:
    #             still_failed.append(idx)
    #
    #     print(f"  Retry {attempt}: recovered {len(failed_indices) - len(still_failed)}/{len(failed_indices)}")
    #     failed_indices = still_failed
    #
    # # --- Random fallback for remaining failures ---
    # for idx in failed_indices:
    #     row = rows[idx]
    #     valid_ids = [row["id_1"], row["id_2"], row["id_3"], row["id_4"]]
    #     a, b = rng.sample(valid_ids, 2)
    #     results[idx] = (a, b, "*", "")
    #
    # if failed_indices:
    #     print(f"  Random fallback used for {len(failed_indices)} rows")
    #
    # # --- Build and save output ---
    # out_rows = []
    # for row, (best, worst, trials, _raw) in zip(rows, results):
    #     out = dict(row)
    #     out["best_pair"] = best
    #     out["worst_pair"] = worst
    #     out["trials"] = trials
    #     out_rows.append(out)
    #
    # output_df = pd.DataFrame(out_rows)
    #
    # output_path = Path(output_file)
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # output_df.to_excel(output_file, index=False, engine="openpyxl")
    #
    # total = len(results)
    # fallbacks = sum(1 for r in results if r[2] == "*")
    # print(f"\nDone. {total} rows processed, {fallbacks} random fallbacks.")
    # print(f"Results saved to: {output_file}")
    #
    # return output_df





if __name__ == "__main__":
    run_time_identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_time_identifier = '2026-02-19_12-45-04'

    #train_mnrl_and_evaluate(run_time_identifier=run_time_identifier)
    #generate_consecutive_pairs('/home/shailu1492/repositories/intellexus-model/sensim/data/merged_kangyur_tengyur_segments_v4.csv', '/home/shailu1492/repositories/intellexus-model/sensim/data/consecutive_segment_pairs_v4')

    # Evaluate models on the 4 sets (A, B, C, D) without any fine-tuning, on external_models.
    evaluate_model_on_sets(run_time_identifier='external_models_bo', use_unicode_columns=True)

    #evaluate_local_llm_on_4_pairs('/home/shailu1492/repositories/intellexus-model/sensim/data/data_4_pairs/test_4_pairs_300.xlsx', '/home/shailu1492/repositories/intellexus-model/sensim/results/bws/qwen/test_4_pairs_300_annotated.xlsx')