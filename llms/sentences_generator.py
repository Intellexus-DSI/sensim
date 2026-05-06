from __future__ import annotations

import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from common_utils import load_dataframe, load_yaml, save_dataframe_single

from .models import SUPPORTED_MODELS, make_chat_model
from .sentences_generator_runner import process_batch_generator, process_single_generator

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMSentencesGenerator:
    # Keys
    _keys: Dict[str, Any]

    # Model & hyperparams
    model: str
    temperature: float
    similarities: List[float]

    # Rate limiter
    use_rate_limiter: bool
    requests_per_second: float
    check_every_n_seconds: int

    # Batching
    batched: bool
    batch_size: int

    # IO
    output_file: Path
    input_file: Path

    # Misc
    debug: bool  # skip model calls and randomly generate outputs

    DEFAULTS = {
        "config_path": None,
        "keys_path": "./keys.yaml",
        "model": "",
        "temperature": 0.7,
        "similarities": [0.9],
        "use_rate_limiter": True,
        "requests_per_second": 1.0,
        "check_every_n_seconds": 60,
        "batched": True,
        "batch_size": 16,
        "input_path": None,
        "output_path": None,
        "debug": False,
    }

    def __init__(
            self,
            config_path: Optional[str] = None,
            keys_path: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            similarities: Optional[List[float]] = None,
            use_rate_limiter: Optional[bool] = None,
            requests_per_second: Optional[float] = None,
            check_every_n_seconds: Optional[float] = None,
            batched: Optional[bool] = None,
            batch_size: Optional[int] = None,
            debug: Optional[bool] = None,
            input_path: Optional[str] = None,
            output_path: Optional[str] = None,
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

        # Misc
        self.debug = bool(choose("debug", debug))

        # Model & hyperparams
        self.model = str(choose("model", model))
        if not self.debug and self.model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of: {list(SUPPORTED_MODELS.keys())}")

        self.temperature = float(choose("temperature", temperature))
        self.similarities = [float(x) for x in choose("similarities", similarities)]
        for s in self.similarities:
            if not (0.0 <= s <= 1.0):
                raise ValueError(f"--similarities values must be in [0,1]. Got {s}")
        LOGGER.info("Using similarity levels: %s", self.similarities)

        # Rate limiter
        self.use_rate_limiter = bool(choose("use_rate_limiter", use_rate_limiter))
        self.requests_per_second = float(choose("requests_per_second", requests_per_second))
        self.check_every_n_seconds = int(choose("check_every_n_seconds", check_every_n_seconds))

        # Batching
        self.batched = bool(choose("batched", batched))
        self.batch_size = int(choose("batch_size", batch_size))

        # IO
        self.input_path = Path(str(choose("input_path", input_path))).expanduser().resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.output_path = Path(str(choose("output_path", output_path))).expanduser().resolve()
        if self.output_path.exists():
            LOGGER.warning("Output file already exists and may be overwritten: %s", self.output_path)

        # Optional: keep raw-progress CSV next to the pretty XLSX
        self._raw_progress_path = self.output_path.with_stem(self.output_path.stem + "_raw")

    def run(self) -> pd.DataFrame:
        LOGGER.info(
            "Starting run | model=%s | temperature=%.2f | batched=%s | debug=%s",
            ("DEBUG" if self.debug else self.model),
            self.temperature,
            self.batched,
            self.debug,
        )

        input_df = load_dataframe(self.input_path)

        anchors = input_df[["ID", "anchor_sentence"]]
        anchors = anchors.drop_duplicates(subset=["anchor_sentence"]).dropna().reset_index(drop=True)
        LOGGER.info("Loaded %d unique anchors from %s", len(anchors), self.input_path)

        output_df = pd.DataFrame(columns=["ID", "anchor_sentence", "candidate", "sim_level", "temperature", "model"])

        chat = None
        if not self.debug:
            chat = make_chat_model(
                self.model,
                self.temperature,
                self._keys,
                use_rate_limiter=self.use_rate_limiter,
                requests_per_second=self.requests_per_second,
                check_every_n_seconds=self.check_every_n_seconds,
            )

        try:
            if (not self.debug) and self.batched:
                total_rows = len(anchors) * len(self.similarities)
                pbar = tqdm(total=total_rows, desc="Processing (batched)", unit="row")

                for chunk_start in range(0, total_rows, self.batch_size):
                    chunk_df = anchors.iloc[chunk_start: chunk_start + self.batch_size]
                    chunk_rows: List[Dict[str, Any]] = chunk_df.to_dict(orient="records")

                    chunk_results = process_batch_generator(
                        chat=chat,
                        rows=chunk_rows,
                        similarities=self.similarities,
                        batch_size=len(chunk_rows),
                        id_key="ID",
                        anchor_key="anchor_sentence",
                    )

                    for out in chunk_results:
                        out["temperature"] = str(self.temperature)
                        out["model"] = ("DEBUG" if self.debug else self.model)

                    if chunk_results:
                        output_df = pd.concat([output_df, pd.DataFrame(chunk_results)], ignore_index=True)

                    # Raw progress checkpoint (CSV)
                    save_dataframe_single(output_df, self._raw_progress_path, exists_ok=True)
                    pbar.update(len(chunk_rows))

                pbar.close()

            else:
                for row in tqdm(
                        anchors.itertuples(index=False),
                        total=len(anchors),
                        desc="Processing (single)",
                        unit="row",
                ):
                    for s in self.similarities:
                        out = process_single_generator(
                            chat=chat,
                            debug=self.debug,
                            similarity=s,
                            row={"ID": row.ID, "anchor_sentence": row.anchor_sentence},
                        )

                        out["temperature"] = str(self.temperature)
                        out["model"] = ("DEBUG" if self.debug else self.model)

                        output_df = pd.concat([output_df, pd.DataFrame([out])], ignore_index=True)

                        # Raw progress checkpoint (CSV)
                        save_dataframe_single(output_df, self._raw_progress_path, exists_ok=True)

        finally:
            # Always checkpoint the raw DF
            save_dataframe_single(output_df, self._raw_progress_path, exists_ok=True)

        LOGGER.info("Generation complete.")

        return output_df

    @staticmethod
    def _parse_cli(argv: Optional[List[str]]) -> Tuple[Dict[str, Any], Optional[str]]:
        if argv is None:
            return {}, None

        p = ArgumentParser(
            description="Generate similar Tibetan sentences via LLM (LangChain).",
            allow_abbrev=False,
        )

        # Config / keys
        p.add_argument("--config", type=str, help="Path to config YAML")
        p.add_argument("--keys", dest="keys_path", type=str, help="Path to keys YAML (API keys)")

        # Model & hyperparams
        p.add_argument("--model", type=str, choices=list(SUPPORTED_MODELS.keys()), help="Model name")
        p.add_argument("--temperature", type=float, help="Sampling temperature")
        p.add_argument(
            "--similarities",
            type=float,
            nargs="+",
            help="Comma-separated similarity levels (e.g. --similarities 0.9 0.8 0.7)",
        )

        # Rate limiter
        p.add_argument(
            "--use-rate-limiter",
            action="store_true",
            default=True,
            dest="use_rate_limiter",
            help="Whether to use rate limiting",
        )
        p.add_argument("--requests-per-second", type=float, dest="requests_per_second", help="Requests per second")
        p.add_argument("--check-every-n-seconds", type=int, dest="check_every_n_seconds", help="Check interval seconds")

        # Batching
        p.add_argument("--batched", action="store_true", default=True, help="Whether to use batching")
        p.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")

        # IO
        p.add_argument("--input-path", type=str, dest="input_path", help="Path to input file")
        p.add_argument("--output-path", type=str, dest="output_path", help="Path to output XLSX")

        # Misc
        p.add_argument("--debug", action="store_true", default=False, help="Debug mode (no API calls)")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)
        maybe_config_path = raw.pop("config", None)

        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
