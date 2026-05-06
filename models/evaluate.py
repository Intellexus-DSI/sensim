import inspect
import numpy as np
from boltons.iterutils import chunked_iter
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_manhattan_distances,
    paired_euclidean_distances,
)
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets
from typing import Any, Dict, List, Optional, Callable

from models.dataset_wrapper import MultiDatasetsWrapper, DatasetWrapper


def _filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by fn (unless fn has **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs

    allowed = {p.name for p in sig.parameters.values()}
    return {k: v for k, v in kwargs.items() if k in allowed}


def _infer_model_kind(model) -> str:
    """
    Decide evaluation mode:
      - 'bi' if model has .encode()
      - 'cross' if model has .predict()
    """
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        return "bi"
    if hasattr(model, "predict") and callable(getattr(model, "predict")):
        return "cross"
    raise TypeError("Model must expose either .encode() (bi-encoder) or .predict() (cross-encoder).")


def evaluate(
        model,
        data: MultiDatasetsWrapper | DatasetWrapper,
        batch_size: int = 32,
        **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Generic evaluator for SentenceTransformer (bi-encoder) and CrossEncoder.

    - Bi-encoder: uses model.encode() and computes correlations on cosine/distances/dot.
    - Cross-encoder: uses model.predict(pairs) and computes correlations on predicted scores.
    """
    if isinstance(data, MultiDatasetsWrapper):
        all_metrics: Dict[str, Dict[str, Any]] = {}
        for ds in data:
            metrics = _evaluate_single(model, ds.dataset, batch_size, **kwargs)
            all_metrics[ds.name] = metrics

        if len(data) > 1:
            merged_ds = concatenate_datasets([ds.dataset for ds in data])
            overall_metrics = _evaluate_single(model, merged_ds, batch_size, **kwargs)
            all_metrics["aggregated"] = overall_metrics
        return all_metrics
    else:
        return {data.name: _evaluate_single(model, data.dataset, batch_size, **kwargs)}


def _evaluate_single(
        model,
        data: Dataset,
        batch_size: int = 32,
        **kwargs,
) -> Dict[str, Any]:
    kind = _infer_model_kind(model)
    if kind == "bi":
        return _evaluate_bi(model, data, batch_size, **kwargs)
    return _evaluate_cross(model, data, batch_size, **kwargs)


def _evaluate_bi(
        model,
        data: Dataset,
        batch_size: int = 32,
        **kwargs,
) -> Dict[str, Any]:
    embeddings1 = []
    embeddings2 = []

    text1 = data["text1"]
    text2 = data["text2"]
    labels = np.asarray(data["label"], dtype=float)

    encode_kwargs = _filter_kwargs(model.encode, kwargs)

    total = (len(text1) + batch_size - 1) // batch_size
    for chunk in tqdm(
            chunked_iter(range(len(text1)), batch_size),
            desc="Encoding batches",
            total=total,
    ):
        batch_text1 = [text1[i] for i in chunk]
        batch_text2 = [text2[i] for i in chunk]

        batch_embeddings1 = model.encode(batch_text1, **encode_kwargs)
        batch_embeddings2 = model.encode(batch_text2, **encode_kwargs)

        embeddings1.append(batch_embeddings1)
        embeddings2.append(batch_embeddings2)

    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)

    embeddings1 = np.nan_to_num(embeddings1, nan=0.0, posinf=1.0, neginf=-1.0)
    embeddings2 = np.nan_to_num(embeddings2, nan=0.0, posinf=1.0, neginf=-1.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=-1.0)

    cosine_scores = 1.0 - paired_cosine_distances(embeddings1, embeddings2)
    manhattan_scores = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_scores = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_scores = np.sum(embeddings1 * embeddings2, axis=1)

    pearson_cosine, pearson_pvalue = pearsonr(labels, cosine_scores)
    spearman_cosine, spearman_pvalue = spearmanr(labels, cosine_scores)
    kendall_cosine, kendall_pvalue = kendalltau(labels, cosine_scores)

    pearson_manhattan, _ = pearsonr(labels, manhattan_scores)
    spearman_manhattan, _ = spearmanr(labels, manhattan_scores)
    kendall_manhattan, _ = kendalltau(labels, manhattan_scores)

    pearson_euclidean, _ = pearsonr(labels, euclidean_scores)
    spearman_euclidean, _ = spearmanr(labels, euclidean_scores)
    kendall_euclidean, _ = kendalltau(labels, euclidean_scores)

    pearson_dot, _ = pearsonr(labels, dot_scores)
    spearman_dot, _ = spearmanr(labels, dot_scores)
    kendall_dot, _ = kendalltau(labels, dot_scores)

    return {
        "pearson_cosine": pearson_cosine,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_cosine,
        "spearman_pvalue": spearman_pvalue,
        "kendall_cosine": kendall_cosine,
        "kendall_pvalue": kendall_pvalue,
        "pearson_manhattan": pearson_manhattan,
        "spearman_manhattan": spearman_manhattan,
        "kendall_manhattan": kendall_manhattan,
        "pearson_euclidean": pearson_euclidean,
        "spearman_euclidean": spearman_euclidean,
        "kendall_euclidean": kendall_euclidean,
        "pearson_dot": pearson_dot,
        "spearman_dot": spearman_dot,
        "kendall_dot": kendall_dot,
    }


def _evaluate_cross(
        model,
        data: Dataset,
        batch_size: int = 32,
        **kwargs,
) -> Dict[str, Any]:
    text1 = data["text1"]
    text2 = data["text2"]
    labels = np.asarray(data["label"], dtype=float)

    predict_kwargs = _filter_kwargs(model.predict, kwargs)

    preds = []
    total = (len(text1) + batch_size - 1) // batch_size
    for chunk in tqdm(
            chunked_iter(range(len(text1)), batch_size),
            desc="Predicting batches",
            total=total,
    ):
        pairs = [(text1[i], text2[i]) for i in chunk]

        if "batch_size" in inspect.signature(model.predict).parameters:
            batch_preds = model.predict(pairs, batch_size=len(pairs), **predict_kwargs)
        else:
            batch_preds = model.predict(pairs, **predict_kwargs)

        preds.append(np.asarray(batch_preds, dtype=float))

    preds = np.concatenate(preds, axis=0)

    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=-1.0)

    pearson_score, pearson_pvalue = pearsonr(labels, preds)
    spearman_score, spearman_pvalue = spearmanr(labels, preds)
    kendall_score, kendall_pvalue = kendalltau(labels, preds)

    return {
        "pearson_cosine": pearson_score,
        "pearson_pvalue": pearson_pvalue,
        "spearman_cosine": spearman_score,
        "spearman_pvalue": spearman_pvalue,
        "kendall_cosine": kendall_score,
        "kendall_pvalue": kendall_pvalue,
    }


def _create_eval_results_single_set(metrics: Optional[Dict[str, Dict[str, Any]]], prefix: str) -> Dict[
    str, Any]:
    """
    Create evaluation result dictionary from single set of metrics.

    Returns dict with formatted metric names.
    """

    def update_results(metric_dict: Optional[Dict[str, Any]], key_prefix: str) -> Dict[str, Any]:
        d = metric_dict or {}
        return {
            f"{key_prefix}_spearman": d.get("spearman_cosine", "-"),
            f"{key_prefix}_spearman_pvalue": d.get("spearman_pvalue", "-"),
            f"{key_prefix}_pearson": d.get("pearson_cosine", "-"),
            f"{key_prefix}_pearson_pvalue": d.get("pearson_pvalue", "-"),
            f"{key_prefix}_kendall": d.get("kendall_cosine", "-"),
            f"{key_prefix}_kendall_pvalue": d.get("kendall_pvalue", "-"),
        }

    results = {}
    if len(metrics) > 1:
        for idx, (name, metric) in enumerate(metrics.items()):
            suffix = "aggregated" if name == "aggregated" else f"{idx}"
            results.update(update_results(metric, f"{prefix}_{suffix}"))
    else:
        values = list(metrics.values())
        metrics = values[0] if values else {}
        results.update(update_results(metrics, prefix))
    return results


def create_eval_results(train_metrics: Optional[Dict[str, Any]],
                        validation_metrics: Optional[Dict[str, Any] | Dict[str, Dict[str, Any]]],
                        test_metrics: Optional[Dict[str, Any] | Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Create evaluation result dictionary from train/val/test metrics.

    Returns dict with formatted metric names.
    """
    metrics = {}
    metrics.update(_create_eval_results_single_set(train_metrics, "train"))
    metrics.update(_create_eval_results_single_set(validation_metrics, "validation"))
    metrics.update(_create_eval_results_single_set(test_metrics, "test"))

    return metrics
