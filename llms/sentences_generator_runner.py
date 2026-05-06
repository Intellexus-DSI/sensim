from __future__ import annotations

import logging
from typing import Dict, List, Any

from tqdm.auto import tqdm

from .prompts_sentence_generator import build_messages

LOGGER = logging.getLogger(__name__)


def _extract_text_from_response(resp: Any) -> str:
    """
    LangChain response normalization.
    Handles:
      - AIMessage with .content as str
      - AIMessage with .content as list[dict] (Gemini style: [{'type':'text','text':...}, ...])
      - plain strings / other objects
    Returns best-effort plain text (no metadata/signatures).
    """
    if resp is None:
        return ""

    content = getattr(resp, "content", resp)

    # Most common: already a string
    if isinstance(content, str):
        return content.strip()

    # Gemini often: list of "parts"
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    texts.append(part.strip())
                continue

            if isinstance(part, dict):
                # Prefer explicit 'text'
                t = part.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
                    continue

                # Sometimes nested or different keys
                if part.get("type") == "text":
                    t2 = part.get("text")
                    if isinstance(t2, str) and t2.strip():
                        texts.append(t2.strip())
                        continue

        return "\n".join(texts).strip()

    # Some providers: dict with 'text'
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t.strip()

    # Fallback
    return str(content).strip()


def process_single_generator(
        chat,
        debug: bool,
        row: Dict[str, str],
        similarity: float,
        *,
        id_key: str = "ID",
        anchor_key: str = "anchor_sentence",
        similarity_key: str = "sim_level",
) -> Dict[str, str]:
    """Process a single anchor via `chat.invoke()`.

    Returns:
        {id_key: anchor_id, anchor_key: anchor, "candidate": response_text, similarity_key: similarity}

    Logic:
        - If debug=True, skip model call and return a debug string.
        - If debug=False, call the model and return its response content.
        - If model call fails, log and raise an exception.
    """
    if id_key not in row:
        raise KeyError(f"Row is missing id_key='{id_key}': {row.keys()}")
    if anchor_key not in row:
        raise KeyError(f"Row is missing anchor_key='{anchor_key}': {row.keys()}")
    anchor_id = str(row[id_key])
    anchor = str(row[anchor_key])
    messages = build_messages(anchor, str(similarity))

    if chat is None and not debug:
        raise RuntimeError("Chat model is not initialized in non-debug mode.")

    if debug:
        last_text = f"Debug mode-id={anchor_id}-anchor={anchor}-similarity={similarity}"
    else:
        try:
            LOGGER.debug("Invoking model for anchor_id=%s", anchor_id)
            resp = chat.invoke(messages)
            last_text = _extract_text_from_response(resp)
        except Exception as e:
            LOGGER.exception("Model call failed on anchor_id=%s with similarity=%s", anchor_id, similarity)
            raise RuntimeError(f"Model call failed on anchor_id={anchor_id} with similarity={similarity}") from e

    return {id_key: str(anchor_id), anchor_key: str(anchor), "candidate": str(last_text) or "",
            similarity_key: str(similarity)}


def process_batch_generator(
        chat,
        batch_size: int,
        rows: List[dict],
        similarities: List[float],
        *,
        id_key: str = "ID",
        anchor_key: str = "anchor_sentence",
        similarity_key: str = "sim_level",
        tqdm_desc: str = "Generating",
) -> List[Dict[str, str]]:
    """Process rows using `chat.batch([...])`.

    Expected row shape (configurable via keys):
        { "<id_key>": "...", "<anchor_key>": "..." }

    Returns a list of dicts (same order as input rows):
        [
          {"ID": "...", "anchor_sentence": "...", "candidate": "...", "similarity": "..." },
          ...
        ]
    """
    if chat is None:
        raise RuntimeError("Chat model is not initialized for batched processing.")

    rows_with_similarities = []
    for row in rows:
        for sim in similarities:
            if id_key not in row:
                raise KeyError(f"Row is missing id_key='{id_key}': {row.keys()}")
            if anchor_key not in row:
                raise KeyError(f"Row is missing anchor_key='{anchor_key}': {row.keys()}")
            rows_with_similarities.append({**row, similarity_key: sim})
    out: List[Dict[str, str]] = []
    total_rows = len(rows_with_similarities)
    if total_rows == 0:
        return out

    batch_size = max(1, min(batch_size, total_rows))

    # tqdm over chunks, progress measured in rows
    with tqdm(total=total_rows, desc=tqdm_desc, unit="row") as pbar:
        for chunk_start in range(0, total_rows, batch_size):
            chunk = rows_with_similarities[chunk_start: chunk_start + batch_size]
            LOGGER.debug(
                "Processing batch chunk rows %d..%d (size=%d)",
                chunk_start,
                chunk_start + len(chunk) - 1,
                len(chunk),
            )

            ids: List[str] = []
            anchors: List[str] = []
            similarities: List[str] = []
            messages_list = []

            for row in chunk:
                sid = str(row[id_key])
                anchor = str(row[anchor_key])
                similarity = str(row[similarity_key])

                ids.append(sid)
                anchors.append(anchor)
                similarities.append(similarity)
                messages_list.append(build_messages(anchor, str(similarity)))

            try:
                responses = chat.batch(messages_list)
            except Exception as e:
                LOGGER.exception(
                    "Batch call failed on chunk rows %d..%d",
                    chunk_start,
                    chunk_start + len(chunk) - 1,
                )
                raise RuntimeError(
                    f"Batch call failed on chunk rows {chunk_start}..{chunk_start + len(chunk) - 1}"
                ) from e

            if len(responses) != len(ids):
                raise RuntimeError(
                    f"Batch response size mismatch: got {len(responses)} responses for {len(ids)} inputs "
                    f"(rows {chunk_start}..{chunk_start + len(chunk) - 1})"
                )

            for sid, anchor, sim, r in zip(ids, anchors, similarities, responses):
                candidate = _extract_text_from_response(r)

                out.append({id_key: sid, anchor_key: anchor, "candidate": candidate, similarity_key: sim})

            pbar.update(len(chunk))

    return out
