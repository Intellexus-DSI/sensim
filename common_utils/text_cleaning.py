# common_utils/text_cleaning.py
from __future__ import annotations

import logging
import re

import pandas as pd

LOGGER = logging.getLogger(__name__)

_BRACKET_RE = re.compile(r"\[.*?\]")


def clean_sentence_series(
        s: pd.Series,
        *,
        remove_brackets: bool = True,
        strip: bool = True,
) -> pd.Series:
    """
    Cleans a pandas Series of sentences safely (no inplace assumptions).

    IMPORTANT:
      - pandas astype(...) returns a new Series. We always assign and return.
      - Works with NaN/None.
    """
    if s is None:
        raise ValueError("clean_sentence_series got None Series")

    out = s.copy()

    out = out.astype(str)

    # Remove bracket patterns like [*] / [abc]
    if remove_brackets:
        out = out.str.replace(_BRACKET_RE, "", regex=True)

    if strip:
        out = out.str.strip()

    return out


def clean_sentences_df(
        df: pd.DataFrame,
        sentence_col: str,
        *,
        drop_empty: bool = True,
        drop_duplicates: bool = True,
        keep: str = "first",
) -> pd.DataFrame:
    """
    Cleans a DF in a consistent way:
      - dropna on sentence_col
      - cast to str
      - remove [..]
      - strip
      - drop empty
      - drop duplicates (by cleaned sentence)
    """
    if sentence_col not in df.columns:
        raise KeyError(f"Missing sentence_col='{sentence_col}'. Available={list(df.columns)}")

    out = df.copy()
    out = out.dropna(subset=[sentence_col]).copy()

    out[sentence_col] = clean_sentence_series(out[sentence_col])

    if drop_empty:
        out = out[out[sentence_col].ne("")].copy()

    if drop_duplicates:
        out = out.drop_duplicates(subset=[sentence_col], keep=keep).reset_index(drop=True)

    return out


def ensure_sentence_id_column(
        df: pd.DataFrame,
        *,
        id_col: str = "ID",
) -> pd.DataFrame:
    """
    Ensures an ID column exists. If missing, creates stable IDs from index.
    """
    out = df.copy()
    if id_col not in out.columns:
        LOGGER.info("ID column missing; generating IDs from index.")
        out[id_col] = out.index.astype(int)
    return out
