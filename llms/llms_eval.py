from __future__ import annotations

import logging
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from training_files_manager import TrainingFilesManager
from common_utils import load_dataframe, load_yaml, save_dataframe_single, SUPPORTED_TABULAR_SUFFIXES

from .models import SUPPORTED_MODELS, make_chat_model, make_local_chat_model
from .bws_runner import process_batch_bws, process_single_bws

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMsEval:
    """Run BWS evaluation via an LLM.

    This class preserves the original behavior:
    - Values are chosen in priority order: explicit args > overrides > CLI > YAML > defaults.
    - Supports debug mode (no API calls).
    - Supports batched / per-row processing.
    """

    # Keys
    _keys: Dict[str, Any]

    # Model & hyperparams
    local_inference: bool
    model: str
    temperature: float
    seed: int

    # Rate limiter
    use_rate_limiter: bool
    requests_per_second: float
    check_every_n_seconds: int

    # Batching
    batched: bool
    batch_size: int
    trials_if_hallucinated: int

    # IO
    _files_manager: TrainingFilesManager

    # Misc
    debug: bool  # skip model calls and randomly generate outputs
    use_unicode: bool  # use Tibetan unicode prompt instead of EWTS prompt

    DEFAULTS = {
        "config_path": None,
        "keys_path": "./keys.yaml",
        "model": "",
        "local_inference": False,
        "temperature": 0,
        "seed": 42,
        "use_rate_limiter": False,
        "requests_per_second": 10.0,
        "check_every_n_seconds": 3,
        "batched": True,
        "llms_prompt_batch_size": 50,
        "trials_if_hallucinated": 1,
        "start_row": 0,
        "debug": False,
        "use_unicode": False,
    }

    def __init__(
            self,
            files_manager: TrainingFilesManager,
            config_path: Optional[str] = None,
            keys_path: Optional[str] = None,
            local_inference: Optional[bool] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            seed: Optional[int] = None,
            trials_if_hallucinated: Optional[int] = None,
            use_rate_limiter: Optional[bool] = None,
            requests_per_second: Optional[float] = None,
            check_every_n_seconds: Optional[float] = None,
            batched: Optional[bool] = None,
            batch_size: Optional[int] = None,
            mode: Optional[str] = None,
            start_row: Optional[int] = None,
            debug: Optional[bool] = None,
            use_unicode: Optional[bool] = None,
            argv: Optional[List[str]] = None,
            **overrides: Any,
    ) -> None:
        cli_args, maybe_config_path = self._parse_cli(argv)
        yaml_args = load_yaml(config_path or maybe_config_path)

        def choose(key: str, given: Any = None) -> Any:
            """Select value in order: explicit arg > overrides > CLI > YAML > default."""
            if given is not None:
                return given
            v = overrides.get(key, None)
            if v is not None:
                return v
            v = cli_args.get(key, None)
            if v is not None:
                return v
            v = yaml_args.get(key, None)
            if v is not None:
                return v
            return self.DEFAULTS[key]

        # Keys
        self._keys = load_yaml(choose("keys_path", keys_path))

        # IO
        self._files_manager = files_manager

        # Misc
        self.debug = bool(choose("debug", debug))
        self.use_unicode = bool(choose("use_unicode", use_unicode))

        # Model & hyperparams
        self.local_inference = bool(choose("local_inference", local_inference))
        self.model = str(choose("model", model))
        # if not self.debug and self.model not in SUPPORTED_MODELS:
        #     raise ValueError(f"model must be one of: {list(SUPPORTED_MODELS.keys())}")

        self.temperature = float(choose("temperature", temperature))
        self.seed = int(choose("seed", seed)) + self._files_manager.current_iteration
        self.trials_if_hallucinated = int(choose("trials_if_hallucinated", trials_if_hallucinated))

        # Rate limiter
        self.use_rate_limiter = bool(choose("use_rate_limiter", use_rate_limiter))
        self.requests_per_second = float(choose("requests_per_second", requests_per_second))
        self.check_every_n_seconds = int(choose("check_every_n_seconds", check_every_n_seconds))

        # Batching
        self.batched = bool(choose("batched", batched))
        if self.local_inference:
            self.batched = False  # force no batching for local inference
        self.batch_size = int(choose("llms_prompt_batch_size", batch_size))


    def run(self, mode: Optional[str] = "complete", four_pairs_file: Optional[Path] = None) -> None:
        if mode not in ["complete", "resume"]:
            raise ValueError("mode must be one of: 'complete', 'resume'")

        LOGGER.info(
            "Starting run | model=%s | mode=%s | temperature=%.2f | batched=%s | debug=%s",
            ("DEBUG" if self.debug else self.model),
            mode,
            self.temperature,
            self.batched,
            self.debug,
        )

        input_file = four_pairs_file if four_pairs_file is not None else self._files_manager.sampled_4_pairs_current
        self._validate_io_file(input_file, is_input=True)

        output_file = self._files_manager.model_similarity_results_current(self.model)
        self._validate_io_file(output_file, is_input=False)

        if mode == "complete" and output_file.exists():
            LOGGER.error("Output exists (complete mode forbids overwrite): %s", output_file)
            raise FileExistsError(f"Output exists (complete mode forbids overwrite): {output_file}")

        if mode == "resume" and not output_file.exists():
            LOGGER.error("Resume mode requires existing output: %s", output_file)
            raise FileNotFoundError(f"Resume mode requires existing output: {output_file}")

        chat = None
        if not self.debug:
            if self.local_inference:
                LOGGER.info("Using local inference for model: %s", self.model)
                chat = make_local_chat_model(self.model, self.temperature)
            else:
                LOGGER.info("Using API calls for model: %s", self.model)
                LOGGER.info(
                    "Rate limiter config: use_rate_limiter=%s, requests_per_second=%s, check_every_n_seconds=%s",
                    self.use_rate_limiter, self.requests_per_second, self.check_every_n_seconds,
                )
                chat = make_chat_model(
                    self.model,
                    self.temperature,
                    self._keys,
                    self.use_rate_limiter,
                    self.requests_per_second,
                    self.check_every_n_seconds,
                )

        rng = random.Random(self.seed)

        input_df = load_dataframe(input_file)
        total = len(input_df)
        LOGGER.info("Loaded %d input rows from %s", total, input_file)

        added_headers = ["best_pair", "worst_pair", "trials"]
        if mode == "complete":
            output_df = pd.DataFrame(columns=list(input_df.columns) + added_headers)
        else:
            output_df = load_dataframe(output_file)
            for header in added_headers:
                if header not in output_df.columns:
                    output_df[header] = []

        required_cols = [
            "pair_1_A", "pair_1_B",
            "pair_2_A", "pair_2_B",
            "pair_3_A", "pair_3_B",
            "pair_4_A", "pair_4_B",
            "id_1", "id_2", "id_3", "id_4",
        ]
        for col in required_cols:
            if col not in input_df.columns:
                LOGGER.error("Input file is missing required column: %s", col)
                raise ValueError(f"Input file is missing required column: {col}")

        start_idx = 0
        if mode == "resume":
            start_idx = len(output_df)
            LOGGER.info("Resuming from index %d", start_idx)

        if start_idx < 0 or start_idx > total:
            raise ValueError(f"start_row {start_idx} out of range for input size {total}")

        rows_to_process = input_df.iloc[start_idx:].to_dict(orient="records")
        LOGGER.info("Processing rows from index %d: %d rows to process", start_idx, len(rows_to_process))

        try:
            if (not self.debug) and self.batched:
                total_rows = len(rows_to_process)
                pbar = tqdm(total=total_rows, desc="Processing (batched)", unit="row")

                for chunk_start in range(0, total_rows, self.batch_size):
                    chunk = rows_to_process[chunk_start: chunk_start + self.batch_size]

                    chunk_results = process_batch_bws(
                        chat=chat,
                        batch_size=len(chunk),
                        trials_if_hallucinated=self.trials_if_hallucinated,
                        rows=chunk,
                        rng=rng,
                        use_unicode=self.use_unicode,
                    )

                    out_rows = []
                    for row, (best, worst, trials, _raw) in zip(chunk, chunk_results):
                        out = dict(row)
                        out["best_pair"] = best
                        out["worst_pair"] = worst
                        out["trials"] = trials
                        out_rows.append(out)

                    if out_rows:
                        output_df = pd.concat([output_df, pd.DataFrame(out_rows)], ignore_index=True)

                    save_dataframe_single(output_df, output_file, exists_ok=True)
                    pbar.update(len(chunk))

                pbar.close()

            else:
                # Non-batched or debug: save after each row
                for row in tqdm(rows_to_process, desc="Processing rows", unit="row"):
                    best, worst, trials, _raw = process_single_bws(
                        chat, self.trials_if_hallucinated, self.debug, row, rng,
                        use_unicode=self.use_unicode,
                    )
                    out = dict(row)
                    out["best_pair"] = best
                    out["worst_pair"] = worst
                    out["trials"] = trials
                    output_df = pd.concat([output_df, pd.DataFrame([out])], ignore_index=True)
                    save_dataframe_single(output_df, output_file, exists_ok=True)
        finally:
            save_dataframe_single(output_df, output_file, exists_ok=True)
            self.seed += 1
            LOGGER.info("Finished writing output to %s", output_file)

    @staticmethod
    def _validate_io_file(path: Path, is_input: bool) -> None:
        suffix = path.suffix.lower()
        if suffix not in [x.lower() for x in SUPPORTED_TABULAR_SUFFIXES]:
            raise ValueError(f"{'Input' if is_input else 'Output'} file has unsupported extension: {suffix}")
        if is_input and not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if not is_input:
            if path.exists():
                LOGGER.warning("Output file already exists and will be overwritten: %s", path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                LOGGER.info("Results will be saved to: %s", path)

    @staticmethod
    def _parse_cli(argv: Optional[List[str]]) -> Tuple[Dict[str, Any], Optional[str]]:
        if argv is None:
            return {}, None

        p = ArgumentParser(description="Run BWS similarity via LLM (LangChain).",
                           allow_abbrev=False, )

        # Config / keys
        p.add_argument("--config", type=str, help="Path to config YAML")
        p.add_argument("--keys", dest="keys_path", type=str, help="Path to keys YAML (API keys)")

        # Model & hyperparams
        p.add_argument("--model", type=str, help="Model name")
        p.add_argument("--temperature", type=float, help="Sampling temperature")
        p.add_argument("--seed", type=int, help="Random seed")
        p.add_argument("--trials-if-hallucinated", type=int, dest="trials_if_hallucinated",
                       help="Number of trials if hallucination is detected")

        # Rate limiter
        p.add_argument("--use-rate-limiter", action="store_true",
                       default=False,
                       dest="use_rate_limiter",
                       help="Whether to use rate limiting")
        p.add_argument("--requests-per-second", type=float, dest="requests_per_second",
                       help="Requests per second for rate limiting")
        p.add_argument("--check-every-n-seconds", type=int, dest="check_every_n_seconds",
                       help="Check every n seconds for rate limiting")

        # Batching
        p.add_argument("--batched", action="store_true", default=True, help="Whether to use batching")
        p.add_argument("--llms-prompt-batch-size", type=int, dest="llms_prompt_batch_size",
                       help="Batch size if batching is used")

        # Misc
        p.add_argument("--debug", action="store_true", default=False, help="Debug mode (no actual API calls)")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)
        maybe_config_path = raw.pop("config", None)

        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
