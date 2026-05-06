"""
Evaluate a non-local LLM (via LangChain) on the gold data file using BWS.

The pipeline mirrors the active-sampling annotation loop:
  1. Gold pairs → BWSFourTupleGenerator → 4-tuples
  2. 4-tuples  → LLMsEval (LangChain API)  → best/worst annotations
  3. Annotations → BWSScorer (Perl)        → BWS similarity scores per pair
  4. BWS scores vs gold scores             → Pearson / Spearman

Usage:
    python -m sub_tasks.eval_gold_data \\
        --config supported_models_config/gemini_2_flash_config.yaml

The given config YAML is the LLM-specific config consumed by LLMsEval
(model, provider, temperature, rate-limiting, etc.).
Infrastructure settings (script paths, sensim_base_dir) are read from
the project-level config.yaml via AppConfig.

Optional CLI overrides:
    --gold-data  <path>   gold .xlsx to evaluate against
                          (default: data/all_gold_pairs_1000_scored.xlsx)
    --output-dir <path>   directory for intermediate and result files
                          (default: results/eval_gold_bws/<timestamp>/)
    --use-unicode         use SentenceA/B_unicode columns instead of EWTS
    --resume              resume an existing partial run from --output-dir
    --debug               skip real LLM calls (random annotations)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_config import AppConfig
from common_utils import load_yaml, concatenate_files, save_dataframe_single
from bws_processing import BWSFourTupleGenerator, BWSScorer
from llms import LLMsEval
from training_files_manager import TrainingFilesManager

LOGGER = logging.getLogger(__name__)
DEFAULT_GOLD_FILENAME = "all_gold_pairs_1000_scored.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def eval_gold_data(
    config_path: str,
    gold_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_unicode: bool = False,
    resume: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """Evaluate a non-local LLM against the gold data using BWS.

    Args:
        config_path: Path to the LLM-specific YAML (e.g. supported_models_config/gemini_2_flash_config.yaml).
        gold_data_path: Override path to the gold .xlsx file.
        output_dir: Directory for all intermediate + results files.
        use_unicode: Score using SentenceA/B_unicode columns instead of EWTS.
        resume: Resume an existing partial run from output_dir.
        debug: Skip real LLM calls and use random annotations.

    Returns:
        Single-row DataFrame with Pearson/Spearman metrics.
    """
    app_config = AppConfig()
    llm_cfg = load_yaml(config_path, must_exist=True)
    model_name: str = llm_cfg.get("model", "")
    if not model_name:
        raise ValueError(f"'model' key missing from config: {config_path}")

    # ── paths — always use absolute paths ────────────────────────────────────
    _fallback_base = str(Path(__file__).resolve().parent.parent)
    sensim_base = str(Path(app_config.get("sensim_base_dir", _fallback_base)).resolve())
    main_config_path = str(Path(sensim_base) / "config.yaml")

    if gold_data_path is None:
        gold_data_path = os.path.join(sensim_base, "data", DEFAULT_GOLD_FILENAME)
    if not Path(gold_data_path).exists():
        raise FileNotFoundError(f"Gold data file not found: {gold_data_path}")

    if output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(sensim_base, "results", "eval_gold_bws", ts)
    output_dir = Path(output_dir).resolve()
    print(f"Output dir: {output_dir}")

    # ── load gold data ────────────────────────────────────────────────────────
    print(f"Loading gold data: {gold_data_path}")
    gold_df = pd.read_excel(gold_data_path, engine="openpyxl")

    col_a = "SentenceA_unicode" if use_unicode else "SentenceA"
    col_b = "SentenceB_unicode" if use_unicode else "SentenceB"
    for col in (col_a, col_b, "ID", "score"):
        if col not in gold_df.columns:
            raise ValueError(f"Gold data missing column '{col}'")

    print(f"  {len(gold_df)} pairs | text: {'unicode' if use_unicode else 'EWTS'} | model: {model_name}")

    # ── set up TrainingFilesManager ───────────────────────────────────────────
    # Pre-place config.yaml so TrainingFilesManager enters resume mode
    # (avoids it overwriting with the project-root config.yaml).
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_dst = output_dir / "config.yaml"
    if not run_config_dst.exists():
        main_config = Path(sensim_base) / "config.yaml"
        shutil.copy2(main_config, run_config_dst)

    fm = TrainingFilesManager(
        models=[model_name],
        parent_dir=output_dir,
        run_datetime=datetime.now(),   # non-None triggers resume check
        start_iteration=0,
        data_dir=os.path.join(sensim_base, "data"),
    )

    # ── write gold pairs as selected_pairs (iteration 0) ─────────────────────
    pairs_df = gold_df[["ID", col_a, col_b, "score"]].rename(
        columns={col_a: "SentenceA", col_b: "SentenceB"}
    )
    selected_pairs_path = fm.selected_pairs_current
    if not (resume and selected_pairs_path.exists()):
        save_dataframe_single(pairs_df, selected_pairs_path, exists_ok=True)
        LOGGER.info("Wrote %d gold pairs → %s", len(pairs_df), selected_pairs_path)

    # ── step 1: generate BWS 4-tuples ─────────────────────────────────────────
    four_pairs_path = fm.sampled_4_pairs_current
    if resume and four_pairs_path.exists():
        print(f"Resuming: found existing 4-tuples ({four_pairs_path.name})")
    else:
        print("Generating BWS 4-tuples ...")
        tuples_gen = BWSFourTupleGenerator(files_manager=fm, config_path=main_config_path)
        tuples_gen.run()

    # ── step 2: annotate with LLM ─────────────────────────────────────────────
    annotations_path = fm.model_similarity_results_current(model_name)
    four_pairs_df = pd.read_excel(four_pairs_path, engine="openpyxl")
    mode = "complete"
    if resume and annotations_path.exists():
        existing_count = len(pd.read_excel(annotations_path, engine="openpyxl"))
        if existing_count >= len(four_pairs_df):
            print(f"Resuming: LLM annotations already complete ({annotations_path.name})")
        else:
            print(f"Resuming: partial LLM annotations ({existing_count}/{len(four_pairs_df)} rows); resuming ...")
            mode = "resume"

    if not (resume and annotations_path.exists() and
            len(pd.read_excel(annotations_path, engine="openpyxl")) >= len(four_pairs_df)):
        print(f"Annotating {len(four_pairs_df)} 4-tuples with {model_name} ...")
        # Resolve keys_path relative to sensim_base so it works from any cwd
        llm_cfg_keys_path = llm_cfg.get("keys_path", "keys.yaml")
        keys_path_abs = str((Path(sensim_base) / llm_cfg_keys_path).resolve())
        llm_eval = LLMsEval(
            files_manager=fm,
            config_path=config_path,
            keys_path=keys_path_abs,
            debug=debug,
            use_unicode=use_unicode,
        )
        llm_eval.run(mode=mode)

    # ── step 3: concatenate (single model → trivial passthrough) ─────────────
    combined_path = fm.llms_4_pair_annotations_current
    if resume and combined_path.exists():
        print(f"Resuming: found existing combined annotations ({combined_path.name})")
    else:
        concatenate_files(fm)

    # ── step 4: BWS scoring ───────────────────────────────────────────────────
    scored_path = fm.llms_pairs_scored_current
    if resume and scored_path.exists():
        print(f"Resuming: found existing BWS scores ({scored_path.name})")
    else:
        print("Running BWS scoring ...")
        bws_scorer = BWSScorer(files_manager=fm, config_path=main_config_path)
        bws_scorer.run()

    # ── step 5: correlate with gold ───────────────────────────────────────────
    scored_df = pd.read_excel(scored_path, engine="openpyxl")
    scored_df["ID"] = scored_df["ID"].astype(str)
    gold_df["ID"] = gold_df["ID"].astype(str)

    merged = pd.merge(
        gold_df[["ID", "score"]].rename(columns={"score": "gold_score"}),
        scored_df[["ID", "score"]].rename(columns={"score": "bws_score"}),
        on="ID",
        how="inner",
    ).dropna(subset=["gold_score", "bws_score"])

    if len(merged) < 2:
        raise RuntimeError(
            f"Too few matching pairs for correlation ({len(merged)}). "
            "Check that IDs in the BWS output match the gold data."
        )

    pearson_r, pearson_p = pearsonr(merged["gold_score"], merged["bws_score"])
    spearman_r, spearman_p = spearmanr(merged["gold_score"], merged["bws_score"])
    kendall_r, kendall_p = kendalltau(merged["gold_score"], merged["bws_score"])

    print(f"\n{'=' * 60}")
    print(f"Model    : {model_name}")
    print(f"Config   : {config_path}")
    print(f"Pairs    : {len(merged)} / {len(gold_df)} scored")
    print(f"Pearson  : {pearson_r:.4f}  (p={pearson_p:.4g})")
    print(f"Spearman : {spearman_r:.4f}  (p={spearman_p:.4g})")
    print(f"Kendall  : {kendall_r:.4f}  (p={kendall_p:.4g})")
    print(f"Output   : {output_dir}")
    print(f"{'=' * 60}")

    results_row = {
        "config_path": config_path,
        "model": model_name,
        "gold_data": Path(gold_data_path).name,
        "text_cols": "unicode" if use_unicode else "EWTS",
        "n_pairs_scored": len(merged),
        "n_pairs_total": len(gold_df),
        "pearson": round(float(pearson_r), 6),
        "pearson_p": round(float(pearson_p), 6),
        "spearman": round(float(spearman_r), 6),
        "spearman_p": round(float(spearman_p), 6),
        "kendall": round(float(kendall_r), 6),
        "kendall_p": round(float(kendall_p), 6),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
    }

    results_csv = output_dir / "eval_gold_results.csv"
    new_row_df = pd.DataFrame([results_row])
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        new_row_df = pd.concat([existing, new_row_df], ignore_index=True)
    new_row_df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")

    return pd.DataFrame([results_row])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a non-local LLM on gold data using BWS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", dest="config_path", required=True,
        help="LLM-specific config YAML (e.g. supported_models_config/gemini_2_flash_lite_config.yaml).",
    )
    p.add_argument(
        "--gold-data", dest="gold_data_path", default=None,
        help=f"Path to the gold .xlsx (default: data/{DEFAULT_GOLD_FILENAME}).",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory for intermediate + result files (default: results/eval_gold_bws/<timestamp>/).",
    )
    p.add_argument(
        "--use-unicode", dest="use_unicode", action="store_true",
        help="Use SentenceA/B_unicode columns instead of EWTS.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an existing partial run from --output-dir.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Skip real LLM calls and use random annotations (for testing).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    eval_gold_data(
        config_path=args.config_path,
        gold_data_path=args.gold_data_path,
        output_dir=args.output_dir,
        use_unicode=args.use_unicode,
        resume=args.resume,
        debug=args.debug,
    )
