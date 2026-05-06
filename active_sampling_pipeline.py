from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import logging
import shutil
import sys
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional

from tqdm.auto import tqdm

from candidates_builder import SimPairsSampler, RandomPairsSampler # PairsSampler
from models import SbertModel
from bws_processing import BWSFourTupleGenerator, BWSScorer, LocalModelsScorer
from common_utils import concatenate_files, load_dataframe, install_global_exception_logging, setup_logging
from llms import LLMsEval, SUPPORTED_MODELS
from training_files_manager import TrainingFilesManager

import os
import yaml
from pathlib import Path

import weave

from app_config import AppConfig

app_config = AppConfig()

LOGGER = logging.getLogger(__name__)


def _build_arg_parser() -> ArgumentParser:
    p = ArgumentParser(description="Run the iterative active sampling pipline.")

    p.add_argument("--number-of-iterations", dest="number_of_iterations",
                   type=int, default=10, help="Number of iterations to run (default: 10)")
    p.add_argument("--resume-time", type=str, default="", dest="resume_time",
                   help="Run datetime to resume (format depends on TrainingFilesManager)")
    p.add_argument("--mock-time", type=str, default="", dest="mock_time",
                   help="Mock run datetime (format depends on TrainingFilesManager)")
    p.add_argument("--run-in-debug-mode", action="store_true", help="Run in debug mode (no calls to LLMs)",
                   dest="run_in_debug_mode")
    # p.add_argument("--segments-dataset", type=str, default="all_segments_1M.xlsx",
    #                help="Path to segments dataset (default: all_segments_1M.xlsx)", dest="segments_dataset")
    p.add_argument("--segments-dataset", type=str, default="merged_kangyur_tengyur_segments_v2.xlsx",
                   help="Path to segments dataset (default: merged_kangyur_tengyur_segments_v2.xlsx)", dest="segments_dataset")


    # -------------------------
    # Logging configuration
    # -------------------------
    p.add_argument("--log-dir", type=str, default="./logs", help="Directory for log files (default: ./logs)")
    p.add_argument("--log-file", type=str, default="active_sampling_pipeline2.log", help="Log filename. If empty, auto-generate one.")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    # Additional arguments
    p.add_argument("--no-console-log", action="store_true", help="Disable console logging (file only)")
    return p


def _parse_args(argv: Optional[List[str]]) -> Namespace:
    p = _build_arg_parser()
    args, _ = p.parse_known_args(argv)
    return args


def _find_models_in_committee(llms_committee):

    matching_configs = []

    for model in llms_committee:
        if model not in SUPPORTED_MODELS:
            LOGGER.warning(f"Model '{model}' from llms_committee is not in SUPPORTED_MODELS. Skipping.")
        else:
            matching_configs.append(SUPPORTED_MODELS[model])

    return matching_configs

def _get_llms_eval(llms_committee, files_manager, argv):
    """
    Load LLMsEval instances based on YAML config files for models in the llms_committee list.
    """
    llms_eval_instances = []
    matching_configs = _find_models_in_committee(llms_committee)

    for config in matching_configs:
        try:
            config_path = config["config_path"]
            llm_eval = LLMsEval(config_path=config_path, files_manager=files_manager, argv=argv)  # Pass actual files_manager and argv as needed
            llms_eval_instances.append(llm_eval)
        except Exception as e:
            print(f"Error initializing LLMsEval for config: {e}")

    return llms_eval_instances

def active_sampling_pipeline(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the active sampling pipeline.

    IMPORTANT: This function refactors structure, logging, and progress reporting,
    while preserving the original pipeline logic and file outputs.
    """
    args = _parse_args(argv)

    base_dir = app_config.get('sensim_base_dir', '.')
    log_dir = app_config.get('log_dir', os.path.join(base_dir, 'logs'))
    log_path = setup_logging(
        log_dir=log_dir,
        log_file=args.log_file,
        log_level=args.log_level,
        console_enabled=(not args.no_console_log),
        name=__name__,
    )
    install_global_exception_logging(LOGGER)

    LOGGER.info("Starting active sampling pipeline. argv=%s", argv)
    LOGGER.info("Log file: %s", log_path)
    resume_time: str = args.resume_time
    resume_mode: bool = bool(resume_time)
    mock_time: str = args.mock_time
    mock_mode: bool = bool(mock_time)
    run_in_debug_mode: bool = bool(args.run_in_debug_mode)

    run_datetime = resume_time if resume_mode else None

    # -------------------------
    # Create TrainingFilesManager first — this copies (or reloads) config.yaml
    # from the run folder so all subsequent config reads use the run's config.
    # no_train_eval logic for validation/test datasets is handled inside
    # TrainingFilesManager after the config is loaded.
    # -------------------------
    _pre_models = ["DEBUG1", "DEBUG2", "DEBUG3"] if run_in_debug_mode else \
        app_config.get('llms_committee', "gemini-2.0-flash")
    _pre_segments = app_config.get("segments_dataset", args.segments_dataset)

    files_manager = TrainingFilesManager(
        models=_pre_models,
        run_datetime=run_datetime,
        segments_dataset=_pre_segments,
    )

    log_path = setup_logging(
        log_dir=str(files_manager.parent_dir),
        log_file=args.log_file,
        log_level=args.log_level,
        console_enabled=(not args.no_console_log),
        name=__name__,
    )
    LOGGER.info("Log redirected to run folder: %s", log_path)

    # -------------------------
    # After TrainingFilesManager, AppConfig is now loaded from the run folder.
    # Re-read all config-dependent values from the (possibly reloaded) config.
    # -------------------------
    number_of_iterations: int = app_config.get('number_of_iterations', args.number_of_iterations)

    if run_in_debug_mode:
        supported_models = ["DEBUG1", "DEBUG2", "DEBUG3"]
    else:
        supported_models = app_config.get('llms_committee', "gemini-2.0-flash")

    segments_dataset = app_config.get("segments_dataset", args.segments_dataset)

    mocked_files_manager: Optional[TrainingFilesManager] = None
    if mock_mode:
        mocked_files_manager = TrainingFilesManager(
            models=supported_models,
            run_datetime=mock_time,
            start_iteration=files_manager.current_iteration,
            segments_dataset=segments_dataset,
        )

    # -------------------------
    # Component initialization
    # -------------------------
    run_config_path = str(files_manager.parent_dir / "config.yaml")
    LOGGER.info("Using run config: %s", run_config_path)
    if app_config.get('random_pair_sampling', False):
        LOGGER.info("Using RandomPairsSampler (random_pair_sampling=true).")
        pairs_sampler = RandomPairsSampler(argv=argv, files_manager=files_manager, config_path=run_config_path)
    else:
        LOGGER.info("Using SimPairsSampler (random_pair_sampling=false).")
        pairs_sampler = SimPairsSampler(argv=argv, files_manager=files_manager, config_path=run_config_path)
    model = SbertModel(config_path=run_config_path, argv=argv, files_manager=files_manager)

    scoring_strategy = app_config.get('scoring_strategy', 'bws')
    LOGGER.info("Scoring strategy: %s", scoring_strategy)

    if run_in_debug_mode:
        gpt_eval = LLMsEval(model="DEBUG1", files_manager=files_manager, debug=True, argv=argv)
        gemini_eval = LLMsEval(model="DEBUG2", files_manager=files_manager, debug=True, argv=argv)
        claude_eval = LLMsEval(model="DEBUG3", files_manager=files_manager, debug=True, argv=argv)
        llms_eval = [gpt_eval, gemini_eval, claude_eval]
        scorer = BWSScorer(argv=argv, files_manager=files_manager)
    elif scoring_strategy == "local_models":
        tuples_generator = None
        llms_eval = []
        scorer = LocalModelsScorer(argv=argv, files_manager=files_manager)
    else:
        tuples_generator = BWSFourTupleGenerator(argv=argv, files_manager=files_manager)
        llms_committee_list: list = app_config.get('llms_committee', "gemini-2.0-flash")
        llms_eval = _get_llms_eval(llms_committee=llms_committee_list, files_manager=files_manager, argv=argv)
        scorer = BWSScorer(argv=argv, files_manager=files_manager)

    # -------------------------
    # Initial evaluation and encoding
    # -------------------------
    if resume_mode and files_manager.current_iteration > 0:
        LOGGER.info("Skipping initial evaluation and encoding (resume mode ON and current iteration > 0).")
        LOGGER.info("Starting from iteration %d.", files_manager.current_iteration)
    elif resume_mode and files_manager.current_iteration == 0 and files_manager.results_file_current.exists() and files_manager.embeddings_current.exists():
        LOGGER.info(
            "Skipping initial evaluation and encoding (resume mode ON, start iteration 0, and existing results).")
        files_manager.increment()
        if mocked_files_manager is not None:
            mocked_files_manager.increment()
    elif app_config.get('random_pair_sampling', False):
        LOGGER.info("Skipping initial evaluation and encoding (random_pair_sampling=true).")
        files_manager.increment()
        if mocked_files_manager is not None:
            mocked_files_manager.increment()
    else:
        LOGGER.info("Running initial evaluation and encoding.")
        model.evaluate_and_export_embeddings()
        files_manager.increment()
        if mocked_files_manager is not None:
            mocked_files_manager.increment()

    if files_manager.results_file_current.exists():
        files_manager.increment()

    # -------------------------
    # Main loop
    # -------------------------
    start_iter = files_manager.current_iteration
    end_iter = number_of_iterations

    loop = tqdm(range(start_iter, end_iter + 1), desc="Progressive learner iterations", unit="it")
    for i in loop:
        loop.set_postfix({"it": f"{i:02d}"})

        if mocked_files_manager is None:
            # Ensure previous embeddings exist (may be missing after format migration or drift cleanup)
            emb_prev = files_manager.embeddings_previous
            if not app_config.get('random_pair_sampling', False) and emb_prev is not None and not emb_prev.exists():
                LOGGER.info("Previous embeddings missing (%s); re-exporting from model checkpoint.", emb_prev)
                model.export_embeddings_to(emb_prev, files_manager.sentences_previous)

            # Sample pairs based on the current model's similarity scores.
            if resume_mode and files_manager.selected_pairs_current.exists():
                LOGGER.info("Skipping pair selection (resume mode ON and existing selected pairs).")
            else:
                LOGGER.info("Sampling pairs for iteration %02d.", i)
                t_sample_start = time.perf_counter()
                pairs_sampler.run()
                sampling_duration = round(time.perf_counter() - t_sample_start, 2)
                model.sampling_duration_seconds = sampling_duration
                model.sampling_batch_count = pairs_sampler.last_batch_count
                model.faiss_distance_mean = pairs_sampler.last_faiss_distance_mean
                model.faiss_distance_std = pairs_sampler.last_faiss_distance_std
                model.sampling_min_dist = pairs_sampler.last_min_dist
                LOGGER.info("Pair sampling took %.2f seconds (%s batches).", sampling_duration, pairs_sampler.last_batch_count)

            if scoring_strategy == "local_models":
                # Score pairs directly with the local models ensemble.
                if resume_mode and files_manager.llms_pairs_scored_current.exists():
                    LOGGER.info("Skipping local-models scoring (resume mode ON and existing file).")
                else:
                    LOGGER.info("Scoring pairs with local models ensemble.")
                    scorer.run()
            else:
                # Generate a distinct set of 4-tuples per LLM model for annotation diversity.
                mode = "complete"
                for m in llms_eval:
                    # Generate (or reuse) per-model tuples file.
                    model_tuples_file = files_manager.model_sampled_4_pairs_current(m.model)
                    if resume_mode and model_tuples_file.exists():
                        LOGGER.info("Skipping 4-pairs generation for %s (resume mode ON and existing file).", m.model)
                    else:
                        LOGGER.info("Generating 4-tuples for model: %s", m.model)
                        tuples_generator.run_for_model(m.model)

                    # Annotate with this model using its own tuples file.
                    if resume_mode and files_manager.model_similarity_results_current(m.model).exists():
                        if len(load_dataframe(files_manager.model_similarity_results_current(m.model))) == \
                                len(load_dataframe(model_tuples_file)):
                            LOGGER.info("Skipping LLM annotation for %s (resume mode ON and existing results).", m.model)
                            continue
                        else:
                            LOGGER.info("Partial results found for %s; resuming annotation.", m.model)
                            mode = "resume"
                    else:
                        mode = "complete"
                    LOGGER.info("Running LLM annotation: %s", m.model)
                    m.run(mode, four_pairs_file=model_tuples_file)
                    mode = "complete"

                # Combine all LLM results into a single file.
                if resume_mode and files_manager.llms_4_pair_annotations_current.exists():
                    LOGGER.info("Skipping LLM outputs concatenation (resume mode ON and existing file).")
                else:
                    LOGGER.info("Concatenating LLM outputs.")
                    concatenate_files(files_manager)

                # Calculate BWS scores.
                if resume_mode and files_manager.llms_pairs_scored_current.exists():
                    LOGGER.info("Skipping BWS scoring (resume mode ON and existing file).")
                else:
                    LOGGER.info("Scoring pairs with BWS.")
                    scorer.run()

        else:
            LOGGER.warning("Mock mode ON: using mocked files (skipping data preparation).")
            dst_dir = files_manager.current_iteration_dir
            dst_dir.mkdir(parents=True, exist_ok=True)

            mocked_files = [
                (mocked_files_manager.selected_pairs_current, files_manager.selected_pairs_current),
                (mocked_files_manager.temp_ids_current, files_manager.temp_ids_current),
                (mocked_files_manager.sampled_4_pairs_current, files_manager.sampled_4_pairs_current),
                *[
                    (
                        mocked_files_manager.model_similarity_results_current(m),
                        files_manager.model_similarity_results_current(m),
                    )
                    for m in supported_models
                ],
                (mocked_files_manager.llms_4_pair_annotations_current, files_manager.llms_4_pair_annotations_current),
                (
                    mocked_files_manager.formatted_llms_4_pair_annotations_current,
                    files_manager.formatted_llms_4_pair_annotations_current,
                ),
                (mocked_files_manager.llms_pairs_scored_current, files_manager.llms_pairs_scored_current),
                (mocked_files_manager.clusters_sizes_current, files_manager.clusters_sizes_current),
                (mocked_files_manager.clusters_stats_current, files_manager.clusters_stats_current),
                (mocked_files_manager.all_pairs_current, files_manager.all_pairs_current),
                (mocked_files_manager.bins_stats_current, files_manager.bins_stats_current),
            ]

            for src, dst in tqdm(mocked_files, desc="Copying mocked files", leave=False):
                if not src.exists():
                    LOGGER.info(f"Skipping missing source file not found: {src}")
                    continue
                    # raise FileNotFoundError(f"Source file not found: {src}")
                if dst.exists():
                    raise FileExistsError(f"Destination file exists: {dst}")
                shutil.copy2(src, dst)

        # Force GPU memory cleanup before training
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for gpu_id in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(gpu_id)
                LOGGER.info("GPU %d memory before training: free %.2f GB / total %.2f GB",
                            gpu_id, free_b / (1024**3), total_b / (1024**3))

        if resume_mode and files_manager.results_file_current.exists():
            LOGGER.info("Skipping model training (resume mode ON and existing results file).")
        else:
            LOGGER.info("Training iteration %02d", i)
            model.supervised_train_iteration()

        if i < number_of_iterations:
            files_manager.increment()
            if mocked_files_manager is not None:
                mocked_files_manager.increment()

    LOGGER.info("Active sampling pipeline completed.")

    # Copy the final merged trainset to results/trainsets/
    merged_trainset_src = files_manager.merged_trainset_path
    if merged_trainset_src.exists():
        trainsets_dir = Path(base_dir) / "results" / "trainsets"
        trainsets_dir.mkdir(parents=True, exist_ok=True)
        dest = trainsets_dir / merged_trainset_src.name
        shutil.copy2(merged_trainset_src, dest)
        LOGGER.info("Final trainset copied to: %s", dest)
    else:
        LOGGER.warning("Merged trainset not found at %s, skipping copy.", merged_trainset_src)
        dest = None

    # Optionally run the eval script with the generated trainset
    if app_config.get('run_eval_after_sampling', False):
        if dest and dest.exists():
            import subprocess
            script_path = Path(base_dir) / "sensim_eval_sbert.sh"
            LOGGER.info("Running eval script: %s %s", script_path, dest)
            subprocess.run(["bash", str(script_path), str(dest)], check=True, cwd=base_dir)
        else:
            LOGGER.warning("run_eval_after_sampling is True but trainset file not found; skipping eval script.")


def main() -> int:
    try:
        active_sampling_pipeline(sys.argv[1:])
        return 0
    except Exception as e:
        # This captures the full stack trace and the specific error message
        LOGGER.error(f"Fatal error of type {type(e).__name__}: {e}")
        LOGGER.exception("Detailed Traceback:")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
