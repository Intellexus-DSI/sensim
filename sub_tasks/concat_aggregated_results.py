
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_config import AppConfig, CONFIG_FILE_NAME


def concat_aggregated_results(results_dir: str, output_filepath: str) -> pd.DataFrame:
    files = sorted(Path(results_dir).glob("*llms_aggregated_sets_results*.csv"))
    if not files:
        raise FileNotFoundError(f"No 'llms_aggregated_sets_results' files found in {results_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.insert(0, "source_file", f.name)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("test_spearman_mean", ascending=False)
    combined.to_csv(output_filepath, index=False)
    print(f"Concatenated {len(files)} files → {output_filepath} ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    app_config = AppConfig(str(Path(__file__).resolve().parent.parent / CONFIG_FILE_NAME))
    sensim_dir = app_config.get("sensim_base_dir") or str(Path(__file__).resolve().parent.parent)

    results_dir = f"{sensim_dir}/results"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filepath = f"{results_dir}/llms_aggregated_all_sets_results_{timestamp}.csv"

    concat_aggregated_results(results_dir, output_filepath)