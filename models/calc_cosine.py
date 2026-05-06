import inspect
import numpy as np
import pandas as pd
from boltons.iterutils import chunked_iter
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances


def _filter_kwargs(fn, kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _infer_model_kind(model) -> str:
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        return "bi"
    if hasattr(model, "predict") and callable(getattr(model, "predict")):
        return "cross"
    raise TypeError("Model must expose either .encode() (bi-encoder) or .predict() (cross-encoder).")


def calc_cosine(model, data, batch_size: int = 32, show_progress: bool = True, **kwargs) -> pd.DataFrame:
    """
    Generic pair scorer.

    - Bi-encoder: returns cosine similarity between embeddings.
    - Cross-encoder: returns model.predict(text1,text2) scores in the same 'cosine' column
      (name kept for backward compatibility with existing pipeline).
    """
    text1 = data["text1"]
    text2 = data["text2"]
    ids = data["ID"]

    kind = _infer_model_kind(model)
    total = (len(text1) + batch_size - 1) // batch_size

    if kind == "bi":
        encode_kwargs = _filter_kwargs(model.encode, kwargs)
        emb1_chunks = []
        emb2_chunks = []

        for chunk in tqdm(chunked_iter(range(len(text1)), batch_size),
                          desc="Encoding batches",
                          total=total,
                          disable=not show_progress):
            batch_text1 = [text1[i] for i in chunk]
            batch_text2 = [text2[i] for i in chunk]
            emb1_chunks.append(model.encode(batch_text1, **encode_kwargs))
            emb2_chunks.append(model.encode(batch_text2, **encode_kwargs))

        emb1 = np.concatenate(emb1_chunks, axis=0)
        emb2 = np.concatenate(emb2_chunks, axis=0)

        scores = 1.0 - paired_cosine_distances(emb1, emb2)

    else:
        predict_kwargs = _filter_kwargs(model.predict, kwargs)
        preds = []

        # If predict supports batch_size, pass it through
        try:
            predict_sig = inspect.signature(model.predict)
        except (TypeError, ValueError):
            predict_sig = None
        supports_bs = predict_sig is not None and ("batch_size" in predict_sig.parameters)

        for chunk in tqdm(chunked_iter(range(len(text1)), batch_size),
                          total=total,
                          disable=not show_progress):
            pairs = [(text1[i], text2[i]) for i in chunk]
            if supports_bs:
                batch_preds = model.predict(pairs, batch_size=batch_size, **predict_kwargs)
            else:
                batch_preds = model.predict(pairs, **predict_kwargs)
            preds.append(np.asarray(batch_preds, dtype=float))

        scores = np.concatenate(preds, axis=0)

    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

    return pd.DataFrame({"ID": ids, "cosine": scores})
