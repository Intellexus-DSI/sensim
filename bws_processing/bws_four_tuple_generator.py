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
class BWSFourTupleGenerator:
    # IO
    _tuples_script: Path
    _files_manager: TrainingFilesManager

    app_config = AppConfig()
    DEFAULTS = {
        "config_path": app_config.get('config_path', None),
        "tuples_script_dir": app_config.get('tuples_script_dir', "/home/shay/Best-Worst-Scaling-Scripts"),
        "tuples_script_file_name": app_config.get('tuples_script_file_name', "generate-BWS-tuples.pl"),
        "n_repeats": app_config.get('bws_n_repeats', 2),
    }

    def __init__(self,
                 # Files manager
                 files_manager: TrainingFilesManager,

                 # Config
                 config_path: Optional[str] = None,

                 # IO
                 tuples_script_dir: Optional[str] = None,
                 tuples_script_file_name: Optional[str] = None,

                 # Number of times to run the tuple generator and combine results
                 n_repeats: Optional[int] = None,

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
        tuples_script_path = os.path.join(
            choose("tuples_script_dir", tuples_script_dir),
            choose("tuples_script_file_name", tuples_script_file_name)
        )
        self._tuples_script = Path(tuples_script_path).expanduser().resolve()
        if self._tuples_script.suffix != '.pl':
            LOGGER.error("Tuples script must be a Perl (.pl) file: %s", self._tuples_script)
            raise ValueError(f"Tuples script must be a Perl (.pl) file: {self._tuples_script}")
        if not self._tuples_script.exists():
            LOGGER.error("Tuples script not found: %s", self._tuples_script)
            raise FileNotFoundError(f"Tuples script not found: {self._tuples_script}")

        # Files manager
        self._files_manager = files_manager

        # Repeat count
        self._n_repeats = int(choose("n_repeats", n_repeats))
        LOGGER.info("BWS tuple generation will run %d time(s)", self._n_repeats)

    def _run_perl_once(self, ids_file: Path) -> List[List[str]]:
        """Run the Perl script once and return the parsed list of ID 4-tuples."""
        cmd = ['perl', str(self._tuples_script.expanduser().resolve()), str(ids_file.expanduser().resolve())]
        result = subprocess.run(cmd, capture_output=True, check=True)

        if result.stderr:
            for line in result.stdout.splitlines():
                LOGGER.info(line.decode('utf-8').strip())
        else:
            LOGGER.warning("No output from BWS tuples script")
            raise RuntimeError("No output from BWS tuples script")

        generated_4_tuples_file = ids_file.with_name(ids_file.name + '.tuples')
        with open(generated_4_tuples_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        id_tuples = []
        for line_num, line in enumerate(lines, 1):
            ids = line.strip().split('\t')
            if len(ids) >= 4:
                id_tuples.append(ids[:4])
            else:
                LOGGER.warning("Line %d has only %d IDs: %s", line_num, len(ids), ids)
                while len(ids) < 4:
                    ids.append('')
                id_tuples.append(ids)

        LOGGER.info("Run produced %d tuples", len(id_tuples))
        return id_tuples

    def run(self) -> None:
        LOGGER.info("Starting run of BWS Four-Tuple Generator")

        input_file = self._files_manager.selected_pairs_current
        if not input_file.suffix in SUPPORTED_TABULAR_SUFFIXES:
            LOGGER.error("Input file must be one of the following types: %s",
                         ', '.join(SUPPORTED_TABULAR_SUFFIXES))
            raise ValueError(f"Input file must be one of the following types: {', '.join(SUPPORTED_TABULAR_SUFFIXES)}")
        LOGGER.info("Input file: %s", str(input_file))
        if not input_file.exists():
            LOGGER.error("Input file not found: %s", input_file)
            raise FileNotFoundError(f"Input file not found: {input_file}")

        ids_file = self._files_manager.temp_ids_current
        if ids_file.exists():
            LOGGER.warning("IDs output file already exists and will be overwritten: %s", str(ids_file))
        LOGGER.info("IDs output file: %s", str(ids_file))

        four_pairs_file = self._files_manager.sampled_4_pairs_current
        if four_pairs_file.exists():
            LOGGER.warning("4-pairs output file already exists and will be overwritten: %s", str(four_pairs_file))
        LOGGER.info("4-pairs output file: %s", str(four_pairs_file))

        df = load_dataframe(input_file)
        required_cols = ['ID', 'SentenceA', 'SentenceB']
        for col in required_cols:
            if col not in df.columns:
                LOGGER.error("Input file is missing required column: %s", col)
                raise ValueError(f"Input file is missing required column: {col}")

        id_list = df['ID'].tolist()

        # Save the list to a text file, one item per line
        with open(ids_file, 'w') as f:
            for item in id_list:
                f.write(str(item) + '\n')

        # Run the Perl script n_repeats times and combine all tuples
        all_id_tuples = []
        for repeat_idx in range(1, self._n_repeats + 1):
            LOGGER.info("Running BWS tuple generation (%d/%d) ...", repeat_idx, self._n_repeats)
            all_id_tuples.extend(self._run_perl_once(ids_file))

        id_tuples = all_id_tuples
        LOGGER.info("Found %d tuples of IDs in total (%d run(s))", len(id_tuples), self._n_repeats)

        # Read the Excel file
        df = load_dataframe(input_file)
        LOGGER.info("Excel DataFrame loaded with %d rows", len(df))

        # Create lookup dictionary for faster access
        LOGGER.info("Creating ID lookup dictionary...")
        id_lookup = {}
        for idx, row in df.iterrows():
            id_val = row['ID']
            id_lookup[id_val] = {
                'sentenceA': row['SentenceA'],
                'sentenceB': row['SentenceB']
            }

        LOGGER.info("Created lookup for %d unique IDs", len(id_lookup))

        # Process each tuple and create result rows
        result_rows = []
        missing_ids = set()

        for tuple_idx, id_tuple in enumerate(id_tuples, 1):
            row_data: Dict[str, str | int] = {'tuple_index': tuple_idx}

            # Process each of the 4 IDs in the tuple
            for pos in range(4):
                id_val = id_tuple[pos] if pos < len(id_tuple) else ''

                # Add ID to row
                row_data[f'id_{pos + 1}'] = id_val

                # Look up sentences
                if id_val and id_val in id_lookup:
                    row_data[f'pair_{pos + 1}_A'] = id_lookup[id_val]['sentenceA']
                    row_data[f'pair_{pos + 1}_B'] = id_lookup[id_val]['sentenceB']
                elif id_val:
                    # ID not found
                    LOGGER.info(
                        f"\nWarning: In tuple_idx [{tuple_idx}], id_tuple [{id_tuple}], pos [{pos}] - id_val {id_val} was not found!")
                    row_data[f'pair_{pos + 1}_A'] = 'NOT_FOUND'
                    row_data[f'pair_{pos + 1}_B'] = 'NOT_FOUND'
                    missing_ids.add(id_val)
                else:
                    # Empty ID
                    LOGGER.info(
                        f"\nWarning: In tuple_idx [{tuple_idx}], id_tuple [{id_tuple}], pos [{pos}] - Empty id_val!")
                    row_data[f'pair_{pos + 1}_A'] = ''
                    row_data[f'pair_{pos + 1}_B'] = ''

            result_rows.append(row_data)

        # Create result DataFrame
        result_df = pd.DataFrame(result_rows)

        # Report missing IDs
        if missing_ids:
            LOGGER.info(f"\nWarning: {len(missing_ids)} IDs not found in Excel file:")
            LOGGER.warning("%d IDs not found in Excel file", len(missing_ids))
            for missing_id in sorted(missing_ids):
                LOGGER.info(f"  - {missing_id}")

        # Save results if output file specified
        save_dataframe_single(result_df, four_pairs_file, exists_ok=True)
        LOGGER.info("Done")

    def run_for_model(self, model_name: str) -> Path:
        """Generate a fresh set of tuples for a specific model and save to a per-model path.

        Returns the path to the saved tuples file.
        """
        LOGGER.info("Generating tuples for model: %s", model_name)

        input_file = self._files_manager.selected_pairs_current
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        ids_file = self._files_manager.temp_ids_current
        df = load_dataframe(input_file)
        id_list = df['ID'].tolist()

        with open(ids_file, 'w') as f:
            for item in id_list:
                f.write(str(item) + '\n')

        all_id_tuples = []
        for repeat_idx in range(1, self._n_repeats + 1):
            LOGGER.info("  [%s] Run %d/%d ...", model_name, repeat_idx, self._n_repeats)
            all_id_tuples.extend(self._run_perl_once(ids_file))

        LOGGER.info("[%s] Total tuples: %d", model_name, len(all_id_tuples))

        id_lookup = {row['ID']: {'sentenceA': row['SentenceA'], 'sentenceB': row['SentenceB']}
                     for _, row in df.iterrows()}

        result_rows = []
        missing_ids = set()
        for tuple_idx, id_tuple in enumerate(all_id_tuples, 1):
            row_data: Dict[str, str | int] = {'tuple_index': tuple_idx}
            for pos in range(4):
                id_val = id_tuple[pos] if pos < len(id_tuple) else ''
                row_data[f'id_{pos + 1}'] = id_val
                if id_val and id_val in id_lookup:
                    row_data[f'pair_{pos + 1}_A'] = id_lookup[id_val]['sentenceA']
                    row_data[f'pair_{pos + 1}_B'] = id_lookup[id_val]['sentenceB']
                elif id_val:
                    row_data[f'pair_{pos + 1}_A'] = 'NOT_FOUND'
                    row_data[f'pair_{pos + 1}_B'] = 'NOT_FOUND'
                    missing_ids.add(id_val)
                else:
                    row_data[f'pair_{pos + 1}_A'] = ''
                    row_data[f'pair_{pos + 1}_B'] = ''
            result_rows.append(row_data)

        if missing_ids:
            LOGGER.warning("[%s] %d IDs not found: %s", model_name, len(missing_ids), sorted(missing_ids))

        result_df = pd.DataFrame(result_rows)
        out_path = self._files_manager.model_sampled_4_pairs_current(model_name)
        save_dataframe_single(result_df, out_path, exists_ok=True)
        LOGGER.info("[%s] Tuples saved to: %s", model_name, out_path)
        return out_path

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
        p.add_argument("--tuples-script-dir",
                       dest="tuples_script_dir",
                       type=str, help="Directory of the tuples script")
        p.add_argument("--tuples-script-file-name",
                       dest="tuples_script_file_name",
                       type=str, help="Tuples script file name")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)

        maybe_config_path = raw.pop("config", None)

        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
