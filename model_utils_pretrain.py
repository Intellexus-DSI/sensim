import numpy as np
import pandas as pd
#from angle_emb import AnglE
from boltons.iterutils import chunked_iter
from pandas.core.interchange.dataframe_protocol import DataFrame
from tqdm import tqdm
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances
)
from scipy.stats import pearsonr, spearmanr, kendalltau

from datasets import Dataset

import text_utils
import utils

EMPTY_METRICS = {
        "pearson_cosine": 0,
        "pearson_pvalue": 0,
        "spearman_cosine": 0,
        "spearman_pvalue": 0,
        "kendall_cosine": 0,
        "kendall_pvalue": 0,
        "pearson_manhattan": 0,
        "spearman_manhattan": 0,
        "pearson_euclidean": 0,
        "spearman_euclidean": 0,
        "pearson_dot": 0,
        "spearman_dot": 0,
    }


def bi_encoder_evaluate(model, data: Dataset, batch_size: int = 32, show_progress: bool = True, **kwargs) -> dict:
    """ Evaluate the model on the given dataset.

    :param model: the model to evaluate.
    :param show_progress: bool, whether to show a progress bar during evaluation.
    :param kwargs: Additional keyword arguments to pass to the `encode` method of the model.

    :return: dict, The evaluation results.
    """
    embeddings1 = []
    embeddings2 = []

    text1 = data['text1']
    text2 = data['text2']
    labels = data['label']

    for chunk in tqdm(chunked_iter(range(len(text1)), batch_size),
                      total=len(text1 )//batch_size,
                      disable=not show_progress):
        batch_text1 = [text1[i] for i in chunk]
        batch_text2 = [text2[i] for i in chunk]

        batch_embeddings1 = model.encode(batch_text1, **kwargs)
        batch_embeddings2 = model.encode(batch_text2, **kwargs)

        embeddings1.append(batch_embeddings1)
        embeddings2.append(batch_embeddings2)

    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)

    embeddings1 = np.nan_to_num(embeddings1, nan=0.0, posinf=1.0, neginf=-1.0)
    embeddings2 = np.nan_to_num(embeddings2, nan=0.0, posinf=1.0, neginf=-1.0)

    cosine_labels = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

    pearson_cosine, pearson_pvalue = pearsonr(labels, cosine_labels)
    spearman_cosine, spearman_pvalue = spearmanr(labels, cosine_labels)
    kendall_cosine, kendall_pvalue = kendalltau(labels, cosine_labels)

    pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    pearson_dot, _ = pearsonr(labels, dot_products)
    spearman_dot, _ = spearmanr(labels, dot_products)

    metrics = {
        "pearson_cosine": pearson_cosine,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_cosine,
        "spearman_pvalue": spearman_pvalue,
        "kendall_cosine": kendall_cosine,
        "kendall_pvalue": kendall_pvalue,
        "pearson_manhattan": pearson_manhattan,
        "spearman_manhattan": spearman_manhattan,
        "pearson_euclidean": pearson_euclidean,
        "spearman_euclidean": spearman_euclidean,
        "pearson_dot": pearson_dot,
        "spearman_dot": spearman_dot,
    }
    return metrics

def cross_encoder_evaluate(model, data: Dataset, batch_size: int = 32, show_progress: bool = True, **kwargs) -> dict:

    sentence_pairs: list[list[str]] = [[text1, text2] for text1, text2 in zip(data['text1'], data['text2'])]
    scores: list[float] = data['label']

    pred_scores = model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=show_progress, batch_size=batch_size)

    pearson_cosine, pearson_pvalue = pearsonr(scores, pred_scores)
    spearman_cosine, spearman_pvalue = spearmanr(scores, pred_scores)

    pearson_manhattan, _ = pearsonr(scores, pred_scores)
    spearman_manhattan, _ = spearmanr(scores, pred_scores)

    pearson_euclidean, _ = pearsonr(scores, pred_scores)
    spearman_euclidean, _ = spearmanr(scores, pred_scores)

    pearson_dot, _ = pearsonr(scores, pred_scores)
    spearman_dot, _ = spearmanr(scores, pred_scores)

    metrics = {
        "pearson_cosine": pearson_cosine,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_cosine,
        "spearman_pvalue": spearman_pvalue,
        "pearson_manhattan": pearson_manhattan,
        "spearman_manhattan": spearman_manhattan,
        "pearson_euclidean": pearson_euclidean,
        "spearman_euclidean": spearman_euclidean,
        "pearson_dot": pearson_dot,
        "spearman_dot": spearman_dot,
    }
    return metrics


def create_eval_result(train_metrics, validation_metrics, test_metrics, test2_metrics=None, prefix=""):
    """
    Create evaluation result dictionary for a single fit.

    If prefix="2", resulting columns will be like "train2_spearman".
    """

    result = {
        # Train
        f"{prefix}train_spearman": train_metrics['spearman_cosine'],
        f"{prefix}train_spearman_pvalue": train_metrics['spearman_pvalue'],
        f"{prefix}train_pearson": train_metrics['pearson_cosine'],
        f"{prefix}train_pearson_pvalue": train_metrics['pearson_pvalue'],
        f"{prefix}train_kendall": train_metrics['kendall_cosine'],
        f"{prefix}train_kendall_pvalue": train_metrics['kendall_pvalue'],

        # Validation
        f"{prefix}validation_spearman": validation_metrics['spearman_cosine'],
        f"{prefix}validation_spearman_pvalue": validation_metrics['spearman_pvalue'],
        f"{prefix}validation_pearson": validation_metrics['pearson_cosine'],
        f"{prefix}validation_pearson_pvalue": validation_metrics['pearson_pvalue'],
        f"{prefix}validation_kendall": validation_metrics['kendall_cosine'],
        f"{prefix}validation_kendall_pvalue": validation_metrics['kendall_pvalue'],

        # Test
        f"{prefix}test_spearman": test_metrics['spearman_cosine'],
        f"{prefix}test_spearman_pvalue": test_metrics['spearman_pvalue'],
        f"{prefix}test_pearson": test_metrics['pearson_cosine'],
        f"{prefix}test_pearson_pvalue": test_metrics['pearson_pvalue'],
        f"{prefix}test_kendall": test_metrics['kendall_cosine'],
        f"{prefix}test_kendall_pvalue": test_metrics['kendall_pvalue'],
    }

    # --- Add Test2 if available ---
    if test2_metrics is not None:
        result.update({
            f"{prefix}test2_spearman": test2_metrics['spearman_cosine'],
            f"{prefix}test2_spearman_pvalue": test2_metrics['spearman_pvalue'],
            f"{prefix}test2_pearson": test2_metrics['pearson_cosine'],
            f"{prefix}test2_pearson_pvalue": test2_metrics['pearson_pvalue'],
            f"{prefix}test2_kendall": test2_metrics['kendall_cosine'],
            f"{prefix}test2_kendall_pvalue": test2_metrics['kendall_pvalue'],
        })

    return result

def create_multi_tests_eval_result(train_metrics, validation_metrics, tests_metrics, prefix=""):

    result = {
        # Train
        f"{prefix}train_spearman": train_metrics['spearman_cosine'],
        f"{prefix}train_spearman_pvalue": train_metrics['spearman_pvalue'],
        f"{prefix}train_pearson": train_metrics['pearson_cosine'],
        f"{prefix}train_pearson_pvalue": train_metrics['pearson_pvalue'],
        f"{prefix}train_kendall": train_metrics['kendall_cosine'],
        f"{prefix}train_kendall_pvalue": train_metrics['kendall_pvalue'],

        # Validation
        f"{prefix}validation_spearman": validation_metrics['spearman_cosine'],
        f"{prefix}validation_spearman_pvalue": validation_metrics['spearman_pvalue'],
        f"{prefix}validation_pearson": validation_metrics['pearson_cosine'],
        f"{prefix}validation_pearson_pvalue": validation_metrics['pearson_pvalue'],
        f"{prefix}validation_kendall": validation_metrics['kendall_cosine'],
        f"{prefix}validation_kendall_pvalue": validation_metrics['kendall_pvalue'],
    }

    # Tests
    for i, test_metrics in enumerate(tests_metrics):
        interation = '' if i == 0 else f'{i}'
        result.update({
            f"{prefix}test{interation}_spearman": test_metrics['spearman_cosine'],
            f"{prefix}test{interation}_spearman_pvalue": test_metrics['spearman_pvalue'],
            f"{prefix}test{interation}_pearson": test_metrics['pearson_cosine'],
            f"{prefix}test{interation}_pearson_pvalue": test_metrics['pearson_pvalue'],
            f"{prefix}test{interation}_kendall": test_metrics['kendall_cosine'],
            f"{prefix}test{interation}_kendall_pvalue": test_metrics['kendall_pvalue'],
        })

    return result

def calc_cosine(model, data: Dataset, batch_size: int = 32, show_progress: bool = True, **kwargs) -> DataFrame:
    embeddings1 = []
    embeddings2 = []

    text1 = data['text1']
    text2 = data['text2']
    ids = data['ID']

    for chunk in tqdm(chunked_iter(range(len(text1)), batch_size),
                      total=len(text1) // batch_size,
                      disable=not show_progress):

        batch_text1 = [text1[i] for i in chunk]
        batch_text2 = [text2[i] for i in chunk]

        batch_embeddings1 = model.encode(batch_text1, **kwargs)
        batch_embeddings2 = model.encode(batch_text2, **kwargs)

        embeddings1.append(batch_embeddings1)
        embeddings2.append(batch_embeddings2)

    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)

    cosine_labels = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    # Create and return the dataframe
    result_df = pd.DataFrame({
        'ID': ids,
        'text1': text1,
        'text2': text2,
        'cosine': cosine_labels
    })

    return result_df


def calculate_pairs_cosine(model, pool_of_pairs_path: str, destination_path: str):
    df_mapping = {'SentenceA': 'text1', 'SentenceB': 'text2', 'ID': 'ID'}

    pool_of_pairs_dataset = utils.get_dataset(pool_of_pairs_path, df_mapping, None)
    cosine_df = calc_cosine(model, pool_of_pairs_dataset)

    # Saving the result.
    original_df = utils.load_dataframe(pool_of_pairs_path)
    original_df = text_utils.clean_sentences(original_df)
    original_df.drop(columns=['cosine', 'bin'], inplace=True, errors='ignore')
    pool_of_pairs_cosine_df = original_df.merge(cosine_df[['ID', 'cosine']], on='ID', how='left')
    utils.save_dataframe(pool_of_pairs_cosine_df, destination_path)
    print('Done saving pairs cosine')
