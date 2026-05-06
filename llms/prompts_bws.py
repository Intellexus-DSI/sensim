from __future__ import annotations

import re
from typing import Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a computational linguist, philologist, and scholar of Classical Tibetan textual traditions.\n"
    "Your task is to analyze text similarity using the BWS method. You will receive 4 pairs of sentences written\n"
    "in Tibetan extended Wylie transliteration. Each pair is labeled P1, P2, P3, or P4. Your job is to identify:\n"
    "- The label of the most similar pair\n"
    "- The label of the least similar pair\n"
    "Focus on semantic and structural alignment.\n"
    "The answer format should be - \n"
    "most similar: <P1/P2/P3/P4>\n"
    "least similar: <P1/P2/P3/P4>\n"
)

SYSTEM_PROMPT_UNICODE = (
    "You are a computational linguist, philologist, and scholar of Classical Tibetan textual traditions.\n"
    "Your task is to analyze text similarity using the BWS method. You will receive 4 pairs of sentences written\n"
    "in Tibetan unicode script transliteration. Each pair is labeled P1, P2, P3, or P4. Your job is to identify:\n"
    "- The label of the most similar pair\n"
    "- The label of the least similar pair\n"
    "Focus on semantic and structural alignment.\n"
    "The answer format should be - \n"
    "most similar: <P1/P2/P3/P4>\n"
    "least similar: <P1/P2/P3/P4>\n"
)

HUMAN_PROMPT_TEMPLATE = (
    "Pair P1: Sentence A - {Sentence_1A} Sentence B - {Sentence_1B}\n"
    "Pair P2: Sentence A - {Sentence_2A} Sentence B - {Sentence_2B}\n"
    "Pair P3: Sentence A - {Sentence_3A} Sentence B - {Sentence_3B}\n"
    "Pair P4: Sentence A - {Sentence_4A} Sentence B - {Sentence_4B}"
)

_MOST_RE = re.compile(r"most\s*similar\s*[:\-]?\s*P([1-4])", re.IGNORECASE)
_LEAST_RE = re.compile(r"least\s*similar\s*[:\-]?\s*P([1-4])", re.IGNORECASE)


def build_messages(row: dict, use_unicode: bool = False) -> list:
    """Build LangChain messages for a single 4-tuple row."""
    msg = HUMAN_PROMPT_TEMPLATE.format(
        Sentence_1A=row["pair_1_A"],
        Sentence_1B=row["pair_1_B"],
        Sentence_2A=row["pair_2_A"],
        Sentence_2B=row["pair_2_B"],
        Sentence_3A=row["pair_3_A"],
        Sentence_3B=row["pair_3_B"],
        Sentence_4A=row["pair_4_A"],
        Sentence_4B=row["pair_4_B"],
    )
    system = SYSTEM_PROMPT_UNICODE if use_unicode else SYSTEM_PROMPT
    return [SystemMessage(content=system), HumanMessage(content=msg)]


def parse_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse most/least labels (1..4) from the model output."""
    most = _MOST_RE.search(text or "")
    least = _LEAST_RE.search(text or "")
    most_label = most.group(1).strip() if most else None
    least_label = least.group(1).strip() if least else None
    return most_label, least_label


def validate_ids(
        most_label: Optional[str],
        least_label: Optional[str],
        valid_ids: list[str],
) -> tuple[bool, Optional[str], Optional[str]]:
    """Validate parsed labels and map them to the corresponding tuple IDs."""
    most_id = None
    least_id = None
    ok = True

    if most_label in {"1", "2", "3", "4"}:
        most_id = valid_ids[int(most_label) - 1]
    else:
        ok = False

    if least_label in {"1", "2", "3", "4"}:
        least_id = valid_ids[int(least_label) - 1]
    else:
        ok = False

    if most_id is not None and least_id is not None and most_id == least_id:
        ok = False

    return ok, most_id, least_id


def row_to_prompt_row(row: dict) -> dict:
    """Validate required columns exist in the row."""
    required = [
        "tuple_index", "id_1", "id_2", "id_3", "id_4",
        "pair_1_A", "pair_1_B", "pair_2_A", "pair_2_B",
        "pair_3_A", "pair_3_B", "pair_4_A", "pair_4_B",
    ]
    for k in required:
        if k not in row:
            raise KeyError(f"Missing required column '{k}'")
    return row
