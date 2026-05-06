from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from argparse import ArgumentParser
import logging

LOGGER = logging.getLogger(__name__)
import subprocess
import pandas as pd
import os

from common_utils import load_yaml, load_dataframe, save_dataframe_single, SUPPORTED_TABULAR_SUFFIXES
from training_files_manager import TrainingFilesManager

from app_config import AppConfig


@dataclass
class BWSScorer:
    # IO
    _scoring_script: Path
    _files_manager: TrainingFilesManager

    app_config = AppConfig()
    DEFAULTS = {
        "config_path": app_config.get('config_path', None),
        "scoring_script_dir": app_config.get('scoring_script_dir', "/home/shay/Best-Worst-Scaling-Scripts"),
        "scoring_script_file_name": app_config.get('scoring_script_file_name', "get-scores-from-BWS-annotations-counting.pl"),
    }

    def __init__(self,
                 # Files manager
                 files_manager: TrainingFilesManager,
                 # Config
                 config_path: Optional[str] = None,

                 # IO
                 scoring_script_dir: Optional[str] = None,
                 scoring_script_file_name: Optional[str] = None,

                 # Command-line arguments
                 argv: Optional[List[str]] = None,

                 # Overrides
                 **overrides: Any, ) -> None:
        cli_args, maybe_config_path = self._parse_cli(argv)
        yaml_args = load_yaml(config_path or maybe_config_path)

        def choose(key: str, given: Any = None):
            """Select value in order: overrides > given_arg > CLI > YAML > default."""
            # explicit argument
            if given is not None:
                return given
            # overrides
            v = overrides.get(key, None)
            if v is not None:
                return v
            # CLI
            v = cli_args.get(key, None)
            if v is not None:
                return v
            # YAML
            v = yaml_args.get(key, None)
            if v is not None:
                return v
            # default
            return self.DEFAULTS[key]

        # IO
        scoring_script_path = os.path.join(
            choose("scoring_script_dir", scoring_script_dir),
            choose("scoring_script_file_name", scoring_script_file_name)
        )
        self._scoring_script = Path(scoring_script_path).expanduser().resolve()
        if not self._scoring_script.exists():
            LOGGER.error("Scoring script not found: %s", self._scoring_script)
            raise FileNotFoundError(f"Scoring script not found: {self._scoring_script}")
        if self._scoring_script.suffix != ".pl":
            LOGGER.error("Scoring script must be a Perl (.pl) file: %s", self._scoring_script)
            raise ValueError(f"Scoring script must be a Perl (.pl) file: {self._scoring_script}")

        self._files_manager = files_manager

    def run(self) -> None:
        LOGGER.info("Starting run of BWS Scorer")

        input_four_pairs_annotations_file = self._files_manager.llms_4_pair_annotations_current
        if input_four_pairs_annotations_file.suffix not in SUPPORTED_TABULAR_SUFFIXES:
            LOGGER.error("Input 4 pairs annotations file must have one of the following extensions: %s",
                         ", ".join(SUPPORTED_TABULAR_SUFFIXES))
            raise ValueError(f"Input 4 pairs annotations file must have one of the following extensions: "
                             f"{', '.join(SUPPORTED_TABULAR_SUFFIXES)}")
        if not input_four_pairs_annotations_file.exists():
            LOGGER.error("Input 4 pairs annotations file not found: %s", input_four_pairs_annotations_file)
            raise FileNotFoundError(f"Input 4 pairs annotations file not found: {input_four_pairs_annotations_file}")
        LOGGER.info("Input 4 pairs annotations file: %s", str(input_four_pairs_annotations_file))

        formatted_four_pairs_annotations_file = self._files_manager.formatted_llms_4_pair_annotations_current
        if formatted_four_pairs_annotations_file.suffix != ".csv":
            LOGGER.error("Formatted 4 pairs annotations file must be a CSV file")
            raise ValueError(f"Formatted 4 pairs annotations file must be a CSV file")
        if formatted_four_pairs_annotations_file.exists():
            LOGGER.warning("Formatted 4 pairs annotations file already exists and will be overwritten: %s",
                           str(formatted_four_pairs_annotations_file))

        input_selected_pairs_file = self._files_manager.selected_pairs_current
        if not input_selected_pairs_file.suffix in SUPPORTED_TABULAR_SUFFIXES:
            LOGGER.error("Input selected pairs file must have one of the following extensions: %s",
                         ", ".join(SUPPORTED_TABULAR_SUFFIXES))
            raise ValueError(f"Input selected pairs file must have one of the following extensions: "
                             f"{', '.join(SUPPORTED_TABULAR_SUFFIXES)}")
        if not input_selected_pairs_file.exists():
            LOGGER.error("Input selected pairs file not found: %s", input_selected_pairs_file)
            raise FileNotFoundError(f"Input selected pairs file not found: {input_selected_pairs_file}")
        LOGGER.info("Input selected pairs file: %s", str(input_selected_pairs_file))

        output_file = self._files_manager.llms_pairs_scored_current
        if not output_file.suffix in SUPPORTED_TABULAR_SUFFIXES:
            LOGGER.error("Output file must have one of the following extensions: %s",
                         ", ".join(SUPPORTED_TABULAR_SUFFIXES))
            raise ValueError(f"Output file must have one of the following extensions: "
                             f"{', '.join(SUPPORTED_TABULAR_SUFFIXES)}")
        if output_file.exists():
            LOGGER.warning("Output file already exists and will be overwritten: %s", str(output_file))
        LOGGER.info("Scores will be saved to: %s", str(output_file))

        # Prepare annotations file for BWS scoring.
        required_cols = ['Item1', 'Item2', 'Item3', 'Item4', 'BestItem', 'WorstItem']
        source_annotated_df = load_dataframe(input_four_pairs_annotations_file)
        cols_to_rename = {"id_1": "Item1", "id_2": "Item2", "id_3": "Item3", "id_4": "Item4", "best_pair": "BestItem",
                          "worst_pair": "WorstItem"}
        source_annotated_df.rename(columns=cols_to_rename, inplace=True)
        for col in required_cols:
            if col not in source_annotated_df.columns:
                LOGGER.error("Missing required column '%s' in input annotations file", col)
                raise ValueError(f"Missing required column '{col}' in input annotations file")
        save_dataframe_single(source_annotated_df[required_cols], formatted_four_pairs_annotations_file)

        # get-scores-from-BWS-annotations-counting.pl
        cmd = ['perl', self._scoring_script.expanduser().resolve(),
               formatted_four_pairs_annotations_file.expanduser().resolve()]

        output_df = pd.DataFrame()
        headers = ['ID', 'raw_score', 'score']
        for col in headers:
            output_df[col] = []

        result = subprocess.run(cmd, capture_output=True, check=True)

        if result.stdout:
            for i, line in enumerate(result.stdout.splitlines()):
                decoded_line = line.decode('utf-8').strip()
                if decoded_line:
                    parts = decoded_line.split('\t')
                    if len(parts) == 2:
                        pair_id, raw_score = parts
                        raw_score = float(raw_score)
                        normalized_score = (raw_score + 1) / 2  # Normalizing score to [0, 1]
                        out = {'ID': pair_id, 'raw_score': raw_score, 'score': normalized_score}
                        output_df = pd.concat([output_df, pd.DataFrame([out])], ignore_index=True)
                    else:
                        LOGGER.warning("Unexpected format in line %d: %s", i + 1, decoded_line)
        else:
            LOGGER.error("No output received from BWS scoring script")
            raise RuntimeError("No output received from BWS scoring script")

        # remove 'raw_score' column for merging
        output_df = output_df[['ID', 'score']]

        # Load selected pairs.
        selected_pairs_df = load_dataframe(input_selected_pairs_file)
        selected_pairs_df.drop(columns=['cosine', 'bin', 'score'], inplace=True, errors='ignore')

        # Merge scores with selected pairs and save.
        output_df['ID'] = output_df['ID'].astype(str)
        selected_pairs_df['ID'] = selected_pairs_df['ID'].astype(str)
        merged_df = pd.merge(selected_pairs_df, output_df, on='ID', how='left')

        missing_ratio = merged_df['score'].isna().mean()

        LOGGER.info("BWS merge missing score ratio: %.2f%%", missing_ratio * 100)

        if missing_ratio > 0.05:
            raise ValueError(
                f"Too many missing scores after BWS merge ({missing_ratio:.1%}). "
                f"ID mismatch likely. Check ID types/format."
            )

        save_dataframe_single(merged_df, output_file, exists_ok=True)

        LOGGER.info("Done")

    @staticmethod
    def _parse_cli(argv: Optional[List[str]]) -> tuple[Dict[str, Any], Optional[str]]:
        if argv is None:
            return {}, None
        p = ArgumentParser(
            description="BWS Four-Tuple Generator",
            allow_abbrev=False,
        )

        # Config
        p.add_argument("--config", type=str, help="Path to config YAML")

        # IO
        p.add_argument("--scoring-script-dir", type=str,
                       dest="scoring_script_dir",
                       help="Directory of the scoring script")
        p.add_argument("--scoring-script-file-name",
                       dest="scoring_script_file_name",
                       type=str, help="Scoring script file name")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)

        maybe_config_path = raw.pop("config", None)

        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
