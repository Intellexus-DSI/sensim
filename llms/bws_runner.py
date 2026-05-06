from __future__ import annotations

import logging
import random
import time
from typing import List, Tuple

from tqdm import tqdm

from .prompts_bws import build_messages, parse_response, row_to_prompt_row, validate_ids

LOGGER = logging.getLogger(__name__)


def _call_with_retry(fn, *args, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
    """Call fn(*args, **kwargs) with exponential-backoff retry on any exception."""
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, max_retries, exc, delay,
            )
            time.sleep(delay)
    raise last_exc


def _random_fallback(valid_ids: List[str], rng: random.Random) -> Tuple[str, str]:
    a, b = rng.sample(valid_ids, 2)
    return a, b


def process_single_bws(
        chat,
        trials_if_hallucinated: int,
        debug: bool,
        row: dict,
        rng: random.Random,
        use_unicode: bool = False,
) -> Tuple[str, str, str, str]:
    """Process a single row via `chat.invoke()`.

    Returns:
        (best_id, worst_id, trials_performed, raw_text)

    Notes:
        - Logic preserved: retry up to `trials_if_hallucinated` times; fallback to random.
        - In debug mode, skips calling the chat model and falls back immediately.
    """
    row_to_prompt_row(row)
    messages = build_messages(row, use_unicode=use_unicode)
    valid_ids = [row["id_1"], row["id_2"], row["id_3"], row["id_4"]]
    max_trials = max(1, trials_if_hallucinated)

    if chat is None and not debug:
        raise RuntimeError("Chat model is not initialized in non-debug mode.")

    last_text = ""
    if not debug:
        for attempt in range(1, max_trials + 1):
            LOGGER.debug(
                "Invoking model for tuple_index=%s (attempt %d/%d)",
                row.get("tuple_index"), attempt, max_trials,
            )
            resp = _call_with_retry(chat.invoke, messages)
            last_text = resp.content if hasattr(resp, "content") else str(resp)

            most_label, least_label = parse_response(last_text or "")
            ok, most_id, least_id = validate_ids(most_label, least_label, valid_ids)
            if ok:
                return most_id, least_id, str(attempt), last_text

            LOGGER.warning(
                "Validation failed for tuple_index=%s (attempt %d/%d). Parsed=%s/%s valid_ids=%s raw=%r",
                row.get("tuple_index"), attempt, max_trials, most_id, least_id, valid_ids, (last_text or "")[:300],
            )

    # exhausted attempts (or debug mode): random fallback
    a, b = _random_fallback(valid_ids, rng)
    if not debug:
        LOGGER.info(
            "tuple_index=%s finalized with random fallback after %d attempts (%s, %s)",
            row.get("tuple_index"), max_trials, a, b,
        )
    return a, b, "*", last_text


def process_batch_bws(
        chat,
        batch_size: int,
        trials_if_hallucinated: int,
        rows: List[dict],
        rng: random.Random,
        use_unicode: bool = False,
) -> List[Tuple[str, str, str, str]]:
    """
    Process rows using `chat.batch([...])`.

    Returns:
        List of (best_id, worst_id, trials_performed, raw_text)

    Logic:
        - One batch call per chunk (attempt #1).
        - For invalid responses, re-batch only the failed items until attempts are exhausted.
        - Remaining invalid items fallback randomly.

    IMPORTANT:
        This function processes ONLY the `rows` list given.
        The caller (run) is responsible for chunking and saving partial progress.
    """
    if chat is None:
        raise RuntimeError("Chat model is not initialized for batched processing.")

    results: List[Tuple[str, str, str, str]] = []
    total_rows = len(rows)
    if total_rows == 0:
        return results

    # If caller passes a "batch_size" larger than rows, cap it
    batch_size = max(1, min(batch_size, total_rows))

    for chunk_start in range(0, total_rows, batch_size):
        chunk = rows[chunk_start: chunk_start + batch_size]
        LOGGER.debug(
            "Processing batch chunk rows %d..%d (size=%d)",
            chunk_start, chunk_start + len(chunk) - 1, len(chunk)
        )

        messages_list = []
        valid_ids_list = []

        for row in chunk:
            row_to_prompt_row(row)
            messages_list.append(build_messages(row, use_unicode=use_unicode))
            valid_ids_list.append([row["id_1"], row["id_2"], row["id_3"], row["id_4"]])

        # Attempt #1: initial batch (with retry on transient errors)
        responses = _call_with_retry(
            chat.batch, messages_list,
            max_retries=5, base_delay=2.0,
        )

        parsed = []  # list of [ok, most_id, least_id, text]
        attempts = [1] * len(chunk)

        for row, valid_ids, resp in zip(chunk, valid_ids_list, responses):
            text = resp.content if hasattr(resp, "content") else str(resp)
            most_label, least_label = parse_response(str(text) or "")
            ok, most_id, least_id = validate_ids(most_label, least_label, valid_ids)
            parsed.append([ok, most_id, least_id, text])

        to_retry = [idx for idx, (ok, *_rest) in enumerate(parsed) if not ok]
        max_trials = max(1, trials_if_hallucinated)
        tries_left = max_trials - 1  # already attempted once

        # Retry only failed items
        while to_retry and tries_left > 0:
            LOGGER.debug("Retrying %d items; tries_left=%d", len(to_retry), tries_left)
            retry_messages = [messages_list[idx] for idx in to_retry]

            try:
                retry_resps = _call_with_retry(
                    chat.batch, retry_messages,
                    max_retries=5, base_delay=2.0,
                )
            except Exception:
                LOGGER.exception("Retry batch failed; will fallback remaining invalid items.")
                break

            for k, idx in enumerate(to_retry):
                attempts[idx] += 1
                text = retry_resps[k].content if hasattr(retry_resps[k], "content") else str(retry_resps[k])

                valid_ids = valid_ids_list[idx]
                row = chunk[idx]

                most_label, least_label = parse_response(str(text) or "")
                ok, most_id, least_id = validate_ids(most_label, least_label, valid_ids)
                parsed[idx] = [ok, most_id, least_id, text]

                if ok:
                    LOGGER.info(
                        "tuple_index=%s recovered on retry (attempts=%d).",
                        row.get("tuple_index"), attempts[idx]
                    )

            to_retry = [idx for idx, (ok, *_rest) in enumerate(parsed) if not ok]
            tries_left -= 1

        # Finalize this chunk: ok => parsed ids, else => random fallback
        for idx, (ok, most_id, least_id, text) in enumerate(parsed):
            row = chunk[idx]
            valid_ids = valid_ids_list[idx]

            if ok:
                results.append((most_id, least_id, str(attempts[idx]), text))
            else:
                a, b = rng.sample(valid_ids, 2)
                LOGGER.info(
                    "tuple_index=%s unresolved after %d attempts; using random fallback (%s, %s)",
                    row.get("tuple_index"), attempts[idx], a, b
                )
                results.append((a, b, "*", text or ""))

    return results
