import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from argparse import ArgumentParser
import logging
from datasets import Dataset
from huggingface_hub import login, HfApi
import wandb
import torch
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, models, losses, evaluation, SentenceTransformerTrainingArguments, \
    SentenceTransformerTrainer

from .dataset_wrapper import DatasetWrapper, MultiDatasetsWrapper
from .evaluate import evaluate, create_eval_results
from .log_results import log_evaluation_results
from .calc_cosine import calc_cosine
from common_utils import load_yaml, load_dataframe, save_dataframe_single, SUPPORTED_TABULAR_SUFFIXES, clean_sentences_df
from training_files_manager import TrainingFilesManager

# import weave
import model_utils

# added the following to enable ModernBERT architecture.
# 1. Force Torch Dynamo to ignore errors and fall back to "eager" mode
import torch._dynamo

from .supported_losses import COSINE_LOSS, COSENT_LOSS, ANGLE_LOSS, CONTRASTIVE_TENSION_LOSS, \
    MULTIPLE_NEGATIVE_RANKING_LOSS, SUPERVISED_LOSSES, UNSUPERVISED_LOSSES

torch._dynamo.config.suppress_errors = True
# 2. Disable the Torch compiler globally
os.environ["TORCHDYNAMO_DISABLE"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XET_NO_PROGRESS"] = "true"

LOGGER = logging.getLogger(__name__)


@dataclass
class SbertModel:
    # Keys
    _hf_token: Optional[str]
    _wandb_token: Optional[str]

    # Model config
    hf_base_model: str

    # Fit config
    _fit_config: Dict[str, Any]
    _pretrained_config: Dict[str, Any]
    pooling_strategy: str
    supervised_loss_type: str
    unsupervised_loss_type: str

    # Embeddings export
    sentence_col: str
    embeddings_col: str

    # IO / logging
    _files_manager: TrainingFilesManager
    _main_results_path: Path
    _reporting_mode: str

    # Datasets
    _validation_ds: MultiDatasetsWrapper
    _test_ds: MultiDatasetsWrapper

    # Misc
    seed: int

    _sbert_instance: Optional[SentenceTransformer] = None

    DEFAULTS = {
        "config_path": None,
        "keys_path": "keys.yaml",
        # "hf_base_model": "Intellexus/mbert-tibetan-continual-wylie-final",
        # "hf_base_model": "Intellexus/IntellexusBert-2.0",
        # "hf_base_model": "Shailu1492/tibetan-modernbert-anchor-positive",
        "hf_base_model": "Shailu1492/tibetan-mbert-v1-anchor-positive",
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "eval_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_strategy": "best",
        "per_device_train_batch_size": 16,  # 32
        "gradient_accumulation_steps": 16,
        "load_best_model_at_end": True,
        "greater_is_better": True,
        "save_total_limit": 2,
        "lr_scheduler_type": "reduce_lr_on_plateau",
        "pooling_strategy": "weightedmean",
        "supervised_loss_type": COSENT_LOSS,
        "unsupervised_loss_type": CONTRASTIVE_TENSION_LOSS,
        "sentence_col": "Segmented_Text_EWTS",
        "embeddings_col": "embeddings",
        "export_core_embeddings_batch_size": 64,
        "track_embedding_drift": False,
        "max_seq_length": None,
        "seed": 42,
    }

    # Setting the float arguments based on the hardware.
    float_args = model_utils.get_float_args()
    DEFAULTS.update(float_args)

    def __init__(self,
                 # Files manager
                 files_manager: TrainingFilesManager,

                 # Config / keys
                 config_path: Optional[str] = None,
                 keys_path: Optional[str] = None,

                 # Model config
                 hf_base_model: Optional[str] = None,
                 max_seq_length: Optional[int] = None,

                 # Fit config
                 num_train_epochs: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 warmup_ratio: Optional[float] = None,
                 eval_strategy: Optional[str] = None,
                 logging_strategy: Optional[str] = None,
                 save_strategy: Optional[str] = None,
                 per_device_train_batch_size: Optional[int] = None,
                 gradient_accumulation_steps: Optional[int] = None,
                 bf16: Optional[bool] = None,
                 load_best_model_at_end: Optional[bool] = None,
                 greater_is_better: Optional[bool] = None,
                 save_total_limit: Optional[int] = None,
                 lr_scheduler_type: Optional[str] = None,
                 pooling_strategy: Optional[str] = None,
                 supervised_loss_type: Optional[str] = None,
                 unsupervised_loss_type: Optional[str] = None,

                 # Embeddings export
                 sentence_col: Optional[str] = None,
                 embeddings_col: str = None,

                 # Misc
                 seed: Optional[int] = None,

                 # Command-line arguments
                 argv: Optional[List[str]] = None,

                 # Overrides
                 **overrides: Any, ) -> None:
        cli_args, maybe_config_path = self._parse_cli(argv)
        yaml_args = load_yaml(config_path or maybe_config_path)

        def choose(key: str, given: Any = None) -> Any:
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
            return self.DEFAULTS.get(key, None)

        # Keys
        _keys = load_yaml(choose("keys_path", keys_path))
        self._hf_token = _keys.get("HF_TOKEN", None)
        if self._hf_token is None:
            LOGGER.warning("HuggingFace token not found in keys; proceeding without authentication.")
        else:
            try:
                login(token=self._hf_token)
                LOGGER.info("Logged into HuggingFace Hub successfully.")
            except Exception as e:
                LOGGER.error("Failed to log into HuggingFace Hub: %s", e)
                raise e

        self._wandb_token = _keys.get("WANDB_TOKEN", None)
        if self._wandb_token:
            try:
                # Log into Weights & Biases
                wandb.login(key=self._wandb_token)
                LOGGER.info("Logged into Weights & Biases successfully. Enabling WandB reporting.")
                self._reporting_mode = 'wandb'
            except Exception as e:
                LOGGER.error("Failed to log into Weights & Biases: %s. Disabling metrics reporting.", e)
                self._reporting_mode = 'none'
        else:
            LOGGER.warning("WandB token not found. Disabling metrics reporting.")
            self._reporting_mode = 'none'

        # Model config
        self.hf_base_model = choose("hf_base_model", hf_base_model)
        self.max_seq_length = choose("max_seq_length", max_seq_length)
        if self.max_seq_length is not None:
            self.max_seq_length = int(self.max_seq_length)

        # Fit config
        self._fit_config = {
            'output_dir': files_manager.model_dir,
            'num_train_epochs': choose("num_train_epochs", num_train_epochs),
            'learning_rate': choose("learning_rate", learning_rate),
            'warmup_ratio': choose("warmup_ratio", warmup_ratio),
            'eval_strategy': choose("eval_strategy", eval_strategy),
            'logging_strategy': choose("logging_strategy", logging_strategy),
            'save_strategy': choose("save_strategy", save_strategy),
            'per_device_train_batch_size': choose("per_device_train_batch_size", per_device_train_batch_size),
            'gradient_accumulation_steps': choose("gradient_accumulation_steps", gradient_accumulation_steps),
            'bf16': choose("bf16", bf16),
            'load_best_model_at_end': choose("load_best_model_at_end", load_best_model_at_end),
            'greater_is_better': choose("greater_is_better", greater_is_better),
            'save_total_limit': choose("save_total_limit", save_total_limit),
            'lr_scheduler_type': choose("lr_scheduler_type", lr_scheduler_type),
            'dataloader_drop_last': True,
            'local_rank': int(os.environ.get("LOCAL_RANK", -1)),
        }
        self.supervised_loss_type = choose("supervised_loss_type", supervised_loss_type)
        self.unsupervised_loss_type = choose("unsupervised_loss_type", unsupervised_loss_type)
        self.pooling_strategy = choose("pooling_strategy", pooling_strategy)

        # no_train_eval mode: skip validation during training
        self._no_train_eval = bool(choose("no_train_eval", None) or False)
        if self._no_train_eval:
            LOGGER.info("no_train_eval mode: disabling load_best_model_at_end and eval during training.")
            self._fit_config['load_best_model_at_end'] = False
            self._fit_config['eval_strategy'] = 'no'
            self._fit_config['save_strategy'] = 'epoch'
            self._fit_config['lr_scheduler_type'] = 'linear'

        # Files manager
        self._files_manager = files_manager
        self._main_results_path = self._check_and_create_path(
            self._files_manager.main_results_path,
            must_exist=False,
            allowed_extensions=SUPPORTED_TABULAR_SUFFIXES,
        )
        self._validation_ds = MultiDatasetsWrapper(self._files_manager.validation_dataset)
        self._test_ds = MultiDatasetsWrapper(self._files_manager.test_dataset)

        # Embeddings export
        self.sentence_col = choose("sentence_col", sentence_col)
        self.embeddings_col = choose("embeddings_col", embeddings_col)
        self.export_core_embeddings_batch_size = int(choose("export_core_embeddings_batch_size", None))
        self.track_embedding_drift = bool(choose("track_embedding_drift", None))
        self._random_pair_sampling = bool(choose("random_pair_sampling", None) or False)

        # Misc
        self._embedding_drift = None  # populated after each embeddings export
        self.sampling_duration_seconds = None  # set externally by the pipeline
        self.sampling_batch_count = None  # set externally by the pipeline
        self.faiss_distance_mean = None
        self.faiss_distance_std = None
        self.sampling_min_dist = None
        self.seed = int(choose("seed", seed)) + self._files_manager.current_iteration

    @staticmethod
    def _check_and_create_path(file_path: Path, must_exist: bool = True,
                               allowed_extensions: Optional[List[str]] = None) -> Path:
        if must_exist and not file_path.exists():
            LOGGER.error("File does not exist: %s", file_path)
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if allowed_extensions and file_path.suffix not in allowed_extensions:
            LOGGER.error("File %s has unsupported extension: %s", file_path, file_path.suffix)
            raise ValueError(f"File {file_path} has unsupported extension: {file_path.suffix}")
        return file_path

    @staticmethod
    def _safe_load_dataframe(dataset_path: Path) -> tuple[pd.DataFrame, Path]:
        input_path = dataset_path.expanduser().resolve()
        if not input_path.exists():
            LOGGER.error("Input file not found: %s", input_path)
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if input_path.suffix not in SUPPORTED_TABULAR_SUFFIXES:
            LOGGER.error("Input file has unsupported extension: %s", input_path.suffix)
            raise ValueError(f"Input file has unsupported extension: {input_path.suffix}")

        return load_dataframe(input_path), input_path

    def _get_dataset(self, dataset_path: Path, use_random_state: Optional[bool] = True,
                     score_required: Optional[bool] = True) -> Dataset:
        df, input_path = self._safe_load_dataframe(dataset_path)

        if score_required and 'score' not in df.columns:
            LOGGER.error("%s file missing 'score' column.", input_path)
            raise ValueError(f"{input_path} file missing 'score' column.")

        df_mapping = {'SentenceA': 'text1', 'SentenceB': 'text2'}
        if score_required and "score" in df.columns:
            df_mapping['score'] = 'label'

        if "ID" in df.columns:
            df_mapping['ID'] = 'ID'

        # Filter columns by keys in df_mapping that exist in df
        cols_to_select = [k for k in df_mapping.keys() if k in df.columns]
        df = df[cols_to_select]
        df.rename(columns=df_mapping, inplace=True)

        # Ensure 'ID' column exists for pool handling later
        if 'ID' not in df.columns:
            df['ID'] = df.index

        if use_random_state and 'label' in df.columns:
            # Only shuffle if it's a labeled training set
            df = df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True)
        dataset = Dataset.from_pandas(df)
        return dataset

    def _get_loss_function(self, loss_name: str):
        """ Get the loss function used by the model during training.

        :param loss_name: The name of the loss function to retrieve.

        :return: The loss function used by the model.
        """
        if loss_name == COSINE_LOSS:
            train_loss = losses.CosineSimilarityLoss(self._sbert_instance)
        elif loss_name == COSENT_LOSS:
            train_loss = losses.CoSENTLoss(self._sbert_instance)
        elif loss_name == ANGLE_LOSS:
            train_loss = losses.AnglELoss(self._sbert_instance)
        elif loss_name == CONTRASTIVE_TENSION_LOSS:
            train_loss = losses.ContrastiveTensionLossInBatchNegatives(self._sbert_instance)
        elif loss_name == MULTIPLE_NEGATIVE_RANKING_LOSS:
            train_loss = losses.MultipleNegativesRankingLoss(self._sbert_instance)
        else:
            LOGGER.error("Unsupported loss type: %s", loss_name)
            raise ValueError(f"Unsupported loss type: {loss_name}")

        return train_loss

    @staticmethod
    def _pick_device_with_info() -> str:
        # Apple Silicon (Mac)
        if torch.backends.mps.is_available():
            LOGGER.info("MPS is available. Using Apple's GPU (mps).")
            LOGGER.info("PyTorch version: %s", torch.__version__)
            LOGGER.info("MPS backend: available (no CUDA-style GPU ids on macOS/MPS)")
            return "mps"

        # NVIDIA CUDA
        if torch.cuda.is_available():
            LOGGER.info("MPS not found. Using CUDA (cuda).")
            LOGGER.info("PyTorch version: %s", torch.__version__)
            LOGGER.info("CUDA compiled version: %s", torch.version.cuda)
            LOGGER.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))

            num_gpus = torch.cuda.device_count()
            LOGGER.info("Number of visible CUDA GPUs: %d", num_gpus)
            _cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _visible_ids = [int(x) for x in _cuda_visible.split(",") if x.strip()] if _cuda_visible else list(range(num_gpus))
            LOGGER.info("Visible GPU ids: %s", _visible_ids)

            for gpu_id, physical_id in zip(range(num_gpus), _visible_ids):
                props = torch.cuda.get_device_properties(gpu_id)
                total_gb = props.total_memory / (1024 ** 3)
                LOGGER.info(
                    "GPU %d: %s | CC %d.%d | VRAM %.2f GB",
                    physical_id, props.name, props.major, props.minor, total_gb
                )

            for gpu_id in range(num_gpus):
                try:
                    free_b, total_b = torch.cuda.mem_get_info(gpu_id)
                    LOGGER.info(
                        "Memory (device %d): free %.2f GB / total %.2f GB",
                        gpu_id, free_b / (1024 ** 3), total_b / (1024 ** 3)
                    )
                except Exception:
                    LOGGER.debug("torch.cuda.mem_get_info() not available for device %d.", gpu_id, exc_info=True)

            LOGGER.info("Returning 'cuda' to use all %d available GPUs.", num_gpus)
            return "cuda"

        # CPU fallback
        LOGGER.info("MPS and CUDA not found. Using CPU.")
        LOGGER.info("PyTorch version: %s", torch.__version__)
        return "cpu"

    def _load_sbert_instance(self) -> SentenceTransformer:
        """Loads SBERT instance from base model or checkpoint."""
        gpu_device = self._pick_device_with_info()

        if self._files_manager.model_dir.exists() and any(self._files_manager.model_dir.iterdir()):
            LOGGER.info("Loading SentenceTransformer from %s", self._files_manager.model_dir.exists())
            model_name_or_path = self._files_manager.model_dir
        else:
            LOGGER.info("Loading SentenceTransformer base model %s", self.hf_base_model)
            model_name_or_path = self.hf_base_model

        transformer_kwargs = {}
        if self.max_seq_length is not None:
            transformer_kwargs["max_seq_length"] = self.max_seq_length
        transformer = models.Transformer(str(model_name_or_path), **transformer_kwargs)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=self.pooling_strategy)

        self._pretrained_config = {
            "model_name_or_path": str(model_name_or_path),
            "pooling_strategy": self.pooling_strategy,
            "loaded_as": "HF Transformer + Pooling",
        }

        # When multiple CUDA GPUs are available, load on CPU so that
        # SentenceTransformerTrainer (HF Trainer) can handle multi-GPU placement.
        # For single-GPU or non-CUDA, pin to the detected device directly.
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            load_device = None  # Trainer will handle device placement
            LOGGER.info("Multiple GPUs detected (%d). Loading model without device pinning for multi-GPU training.", num_gpus)
        else:
            load_device = gpu_device

        self._sbert_instance = SentenceTransformer(modules=[transformer, pooling], device=load_device)

        return self._sbert_instance

    @torch.no_grad()
    def _snapshot_params(self):
        # store on CPU to avoid GPU memory growth
        return {n: p.detach().float().cpu().clone()
                for n, p in self._sbert_instance.named_parameters() if p.requires_grad}

    def _train_with_diff_stats(self, train_ds: DatasetWrapper, validation_ds: MultiDatasetsWrapper,
                               train_loss_name: str, ) -> \
            Dict[str, Any]:
        before = self._snapshot_params()
        self._train(train_ds, validation_ds, train_loss_name=train_loss_name)
        after = self._snapshot_params()
        total_l2 = 0.0
        total_norm_before = 0.0
        max_abs = 0.0
        max_name = None

        for n, b in before.items():
            a = after[n]
            d = a - b
            l2 = d.norm().item()
            total_l2 += l2
            total_norm_before += b.norm().item()
            m = d.abs().max().item()
            if m > max_abs:
                max_abs = m
                max_name = n

        diff_stats = {
            "delta_l2_sum": total_l2,
            "rel_delta_l2": total_l2 / (total_norm_before + 1e-12),
            "max_abs_delta": max_abs,
            "max_abs_param": max_name,
        }
        LOGGER.info("Iteration %d training parameter changes: %s", self._files_manager.current_iteration,
                    diff_stats)
        return diff_stats

    def _train(self, train_ds: DatasetWrapper, validation_ds: MultiDatasetsWrapper, train_loss_name: str) -> None:
        LOGGER.info(f"Training model.")

        def make_evaluator(ds: DatasetWrapper):
            _ds = ds.dataset
            _name = ds.name
            return evaluation.EmbeddingSimilarityEvaluator(
                _ds["text1"],
                _ds["text2"],
                [float(x) for x in _ds["label"]],
                name=_name,
                main_similarity="cosine",
            )

        if validation_ds is None or len(validation_ds) == 0:
            LOGGER.warning("No validation dataset provided; training without evaluation.")
            evaluator = None
            eval_dataset = None
            metric_for_best = None
            self._fit_config["eval_strategy"] = "no"  # Hard override to avoid errors.
            self._fit_config["lr_scheduler_type"] = "linear"  # Hard override to avoid errors.
            self._fit_config["metric_for_best_model"] = "loss"  # Hard override to avoid errors.
            self._fit_config["save_strategy"] = "epoch"  # Hard override to avoid errors.

        elif len(validation_ds) > 1:
            evaluators = [make_evaluator(ds) for ds in validation_ds]
            eval_dataset = {ds.name: ds.dataset for ds in validation_ds}
            evaluator = evaluation.SequentialEvaluator(evaluators)
            metric_for_best = "eval_sequential_score"
        else:
            vds = validation_ds.first
            evaluator = make_evaluator(vds)
            eval_dataset = vds.dataset
            metric_for_best = f"eval_{vds.name}_spearman_cosine"

        self._fit_config["metric_for_best_model"] = metric_for_best
        self._fit_config["greater_is_better"] = True

        LOGGER.info("metric_for_best_model=%s (greater_is_better=%s)", metric_for_best, True)

        # Precompute max_steps to avoid "dataloader does not have a length" error
        # when Accelerate wraps the dataloader without __len__ in multi-GPU mode.
        import math as _math
        _n_samples = len(train_ds.dataset)
        _n_gpus = max(torch.cuda.device_count(), 1)
        _per_device_batch = self._fit_config.get("per_device_train_batch_size", 32)
        _grad_accum = self._fit_config.get("gradient_accumulation_steps", 1)
        _n_epochs = self._fit_config.get("num_train_epochs", 1)
        _drop_last = self._fit_config.get("dataloader_drop_last", False)
        _effective_batch = _per_device_batch * _n_gpus * _grad_accum
        _steps_per_epoch = (_n_samples // _effective_batch) if _drop_last else _math.ceil(_n_samples / _effective_batch)
        _max_steps = max(_steps_per_epoch, 1) * int(_n_epochs)
        self._fit_config["max_steps"] = _max_steps
        LOGGER.info("Computed max_steps=%d (n_samples=%d, effective_batch=%d, epochs=%s)",
                    _max_steps, _n_samples, _effective_batch, _n_epochs)

        sentence_transformer_training_arguments = SentenceTransformerTrainingArguments(**self._fit_config)
        train_loss = self._get_loss_function(train_loss_name)

        # === PATCH PATCH ===
        # Manually add the missing attribute to prevent the crash
        if not hasattr(sentence_transformer_training_arguments, "save_safetensors"):
            sentence_transformer_training_arguments.save_safetensors = False
        # ======================

        trainer = SentenceTransformerTrainer(
            model=self._sbert_instance,
            train_dataset=train_ds.dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=evaluator,
            args=sentence_transformer_training_arguments,
        )
        trainer.train()
        self._sbert_instance.save(str(self._files_manager.model_dir))
        LOGGER.info("Model training complete. Model saved to %s", self._files_manager.model_dir)

    def _evaluate(self, train_ds: Optional[DatasetWrapper] = None,
                  validation_ds: MultiDatasetsWrapper = None,
                  test_ds: MultiDatasetsWrapper = None,
                  loss_type: Optional[str] = None,
                  train_diff_stats: Optional[Dict[str, Any]] = None) -> None:

        LOGGER.info('Evaluating.')
        log_path = self._check_and_create_path(self._files_manager.results_file_current, must_exist=False,
                                               allowed_extensions=SUPPORTED_TABULAR_SUFFIXES)

        def process_evaluation(ds: Optional[MultiDatasetsWrapper | DatasetWrapper], dataset_type: str) -> \
                Optional[Dict[str, Dict[str, Any]]]:
            if ds is not None:
                metrics = evaluate(self._sbert_instance, ds, batch_size=4, **{'show_progress_bar': False})
                for idx, (name, metrics_item) in enumerate(metrics.items()):
                    LOGGER.info('%s metrics for dataset %d (%s): %s', dataset_type, idx + 1, name,
                                metrics_item)
                    LOGGER.info('%s set spearman_cosine for dataset %d (%s): %s', dataset_type, idx + 1, name,
                                metrics_item['spearman_cosine'])
                    return metrics
            return {}

        train_metrics = process_evaluation(train_ds, 'Train')
        validation_metrics = process_evaluation(validation_ds, 'Validation')
        test_metrics = process_evaluation(test_ds, 'Test')
        LOGGER.info('Done evaluating')

        # Logging.
        eval_results = create_eval_results(train_metrics, validation_metrics, test_metrics)

        def strify_names(names: Optional[str | List[str]]) -> str:
            if names is None:
                return '-'
            if isinstance(names, list):
                return ', '.join(names)
            return names

        run_settings = {"model": f"SentenceTransformer", "manual": "1",
                        "loss_type": loss_type or "-",
                        "iteration": self._files_manager.current_iteration,
                        'train_dataset': strify_names(train_ds.name) if train_ds is not None else '-',
                        'validation_dataset': strify_names(validation_ds),
                        'test_dataset': strify_names(test_ds),
                        'random_state': self.seed,
                        }

        if self.sampling_duration_seconds is not None:
            run_settings['sampling_duration_seconds'] = self.sampling_duration_seconds
        if self.sampling_batch_count is not None:
            run_settings['sampling_batch_count'] = self.sampling_batch_count
        if self.faiss_distance_mean is not None:
            run_settings['faiss_distance_mean'] = self.faiss_distance_mean
        if self.faiss_distance_std is not None:
            run_settings['faiss_distance_std'] = self.faiss_distance_std
        if self.sampling_min_dist is not None:
            run_settings['sampling_min_dist'] = self.sampling_min_dist

        run_settings.update(self._pretrained_config)
        run_settings.update(self._fit_config)
        if train_diff_stats is not None:
            run_settings.update(train_diff_stats)

        log_evaluation_results(log_path=log_path, settings=run_settings, results=eval_results)
        LOGGER.info(f'Evaluation results saved to {log_path}')
        log_evaluation_results(log_path=self._main_results_path, settings=run_settings, results=eval_results)
        LOGGER.info(f'Evaluation results saved to {self._main_results_path}')

    def _get_unlabeled_dataset_for_unsup_pretrain(
            self,
            dataset_path: Path,
            use_random_state: bool = True,
    ) -> Dataset:
        df, input_path = self._safe_load_dataframe(dataset_path)

        if self.sentence_col not in df.columns:
            LOGGER.error(
                "%s missing sentence column %s. Available: %s",
                input_path, self.sentence_col, list(df.columns)
            )
            raise ValueError(
                f"{input_path} missing sentence column {self.sentence_col}. Available: {list(df.columns)}"
            )

        # Keep only what we need
        df = df[[self.sentence_col]].copy()

        # Shared cleaning
        df = clean_sentences_df(df, self.sentence_col, drop_empty=True, drop_duplicates=True)

        if len(df) == 0:
            LOGGER.error("No sentences found in %s after cleaning.", input_path)
            raise ValueError(f"No sentences found in {input_path} after cleaning.")

        out_df = pd.DataFrame({
            "text1": df[self.sentence_col].astype(str),
            "text2": df[self.sentence_col].astype(str),
        })

        if use_random_state:
            out_df = out_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return Dataset.from_pandas(out_df, preserve_index=False)

    def unsupervised_train(self, sample_size: Optional[int] = None) -> None:
        LOGGER.info("Unsupervised train.")

        self._sbert_instance = self._load_sbert_instance()

        validation_ds = self._validation_ds.with_factory(self._get_dataset)
        test_ds = self._test_ds.with_factory(self._get_dataset)

        if sample_size is not None:
            LOGGER.info("Sampling %d rows from unlabeled dataset for unsupervised pretraining.", sample_size)
            core_path = self._check_and_create_path(
                self._files_manager.core_dataset,
                must_exist=True,
                allowed_extensions=SUPPORTED_TABULAR_SUFFIXES,
            )
            full_ds = load_dataframe(core_path)
            if len(full_ds) < sample_size:
                LOGGER.warning(
                    "Requested sample size %d is larger than dataset size %d. Using full dataset.",
                    sample_size, len(full_ds)
                )
                sample_size = len(full_ds)
            sampled_df = full_ds.sample(n=sample_size, random_state=self.seed).reset_index(drop=True)
            sampled_path = self._check_and_create_path(
                self._files_manager.unlabeled_sampled_sentences_current,
                must_exist=False,
                allowed_extensions=SUPPORTED_TABULAR_SUFFIXES,
            )
            save_dataframe_single(sampled_df, sampled_path, exists_ok=True)
            unlabeled_path = sampled_path
            LOGGER.info("Sampled dataset saved to %s", sampled_path)
        else:
            unlabeled_path = self._files_manager.core_dataset

        unlabeled_path = self._check_and_create_path(
            unlabeled_path,
            must_exist=True,
            allowed_extensions=SUPPORTED_TABULAR_SUFFIXES,
        )

        train_ds = DatasetWrapper(unlabeled_path, None, self._get_unlabeled_dataset_for_unsup_pretrain)
        training_validation_ds = validation_ds

        if self.unsupervised_loss_type == CONTRASTIVE_TENSION_LOSS:
            training_validation_ds = None

        train_diff_stats = self._train_with_diff_stats(
            train_ds=train_ds,
            validation_ds=training_validation_ds,
            train_loss_name=self.unsupervised_loss_type,
        )

        self.evaluate_and_export_embeddings(
            train_ds=None,
            validation_ds=validation_ds,
            test_ds=test_ds,
            loss_type=self.unsupervised_loss_type,
            train_diff_stats=train_diff_stats,
        )
        self.seed += 1
        LOGGER.info("Done unsupervised training.")

    def publish_model(self, repo_id: str, commit_message: Optional[str] = "Upload unsupervised trained model") -> None:
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)

        api.upload_folder(
            folder_path=self._files_manager.model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        LOGGER.info(f"Uploaded to https://huggingface.co/{repo_id}")

    def supervised_train_iteration(self) -> None:
        """Subsequent iteration training logic."""
        iteration = self._files_manager.current_iteration
        LOGGER.info(f'Training iteration {iteration}.')
        self._sbert_instance = self._load_sbert_instance()

        # Build cumulative trainset from all iterations.
        self._files_manager.build_merged_trainset()

        # Train only on the current new set.
        train_ds_path = self._check_and_create_path(
            self._files_manager.llms_pairs_scored_current,
            must_exist=True,
            allowed_extensions=SUPPORTED_TABULAR_SUFFIXES,
        )
        train_ds = DatasetWrapper(train_ds_path, None, self._get_dataset)
        validation_ds = self._validation_ds.with_factory(self._get_dataset)
        test_ds = self._test_ds.with_factory(self._get_dataset)
        LOGGER.info('Training iteration %s on dataset: %s', iteration, train_ds.name)
        train_diff_stats = self._train_with_diff_stats(train_ds, validation_ds, self.supervised_loss_type)
        self._evaluate_pairs(self._files_manager.selected_pairs_current)
        self.evaluate_and_export_embeddings(
            train_ds,
            validation_ds,
            test_ds,
            self.supervised_loss_type,
            train_diff_stats=train_diff_stats,
            skip_embeddings_export=self._random_pair_sampling,
        )

        self.seed = self.seed + 1
        LOGGER.info('Training complete.')

    def evaluate_and_export_embeddings(self, train_ds: Optional[DatasetWrapper] = None,
                                       validation_ds: Optional[MultiDatasetsWrapper] = None,
                                       test_ds: Optional[MultiDatasetsWrapper] = None,
                                       loss_type: Optional[str] = None,
                                       train_diff_stats: Optional[Dict[str, Any]] = None,
                                       skip_embeddings_export: bool = False) -> None:
        """Evaluation and embeddings export logic."""
        LOGGER.info('Evaluating and exporting embeddings.')
        if self._sbert_instance is None:
            self._sbert_instance = self._load_sbert_instance()
        if validation_ds is None:
            validation_ds = self._validation_ds.with_factory(self._get_dataset)
        if test_ds is None:
            test_ds = self._test_ds.with_factory(self._get_dataset)
        self._evaluate(
            train_ds,
            validation_ds,
            test_ds,
            loss_type=loss_type,
            train_diff_stats=train_diff_stats,
        )
        if skip_embeddings_export:
            LOGGER.info("Skipping core embeddings export (skip_embeddings_export=True).")
        else:
            self._export_core_embeddings()

        # Log embedding drift to results files (computed during export)
        if self._embedding_drift is not None:
            drift_col = {"embedding_drift": self._embedding_drift}
            self._append_column_to_results(self._files_manager.results_file_current, drift_col)
            self._append_column_to_results(self._main_results_path, drift_col)

    def _append_column_to_results(self, path: Path, columns: Dict[str, Any]) -> None:
        """Update the last row of a results file with additional columns."""
        try:
            if not path.exists():
                return
            df = load_dataframe(path)
            if df.empty:
                return
            for col, val in columns.items():
                df.loc[df.index[-1], col] = val
            save_dataframe_single(df, path, exists_ok=True)
            LOGGER.info("Updated %s with %s", path, columns)
        except Exception as e:
            LOGGER.warning("Failed to update results file %s with drift: %s", path, e)

    def export_embeddings_to(self, emb_path: Path, sent_path: Path) -> None:
        """Re-export embeddings and sentences to arbitrary paths using the current model."""
        if self._sbert_instance is None:
            self._sbert_instance = self._load_sbert_instance()

        core_df = load_dataframe(self._files_manager.core_dataset)
        if self.sentence_col not in core_df.columns:
            raise KeyError(f"core_dataset missing column '{self.sentence_col}'")

        sentences = clean_sentences_df(core_df, self.sentence_col, drop_empty=True, drop_duplicates=True)[
            self.sentence_col]

        if len(sentences) == 0:
            raise ValueError("No sentences found in core_dataset after cleaning.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOGGER.info("export_embeddings_to: Encoding sentences with batch size %d", self.export_core_embeddings_batch_size)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            pool = self._sbert_instance.start_multi_process_pool()
            try:
                embeddings = self._sbert_instance.encode_multi_process(
                    sentences.tolist(), pool,
                    batch_size=self.export_core_embeddings_batch_size,
                    show_progress_bar=True,
                ).astype(np.float32, copy=False)
            finally:
                self._sbert_instance.stop_multi_process_pool(pool)
        else:
            embeddings = self._sbert_instance.encode(
                sentences.tolist(),
                batch_size=self.export_core_embeddings_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            ).astype(np.float32, copy=False)

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        np.save(sent_path, np.array(sentences.tolist(), dtype=object))
        LOGGER.info("Re-exported embeddings -> %s, sentences -> %s", emb_path, sent_path)

    def _export_core_embeddings(self) -> None:
        """
        Encodes *all unique* core sentences and saves them to an embeddings file
        that the FAISS sampler can read.

        Output format:
          - column: 'ID' (string)
          - column: 'sentence' (string)
          - column: 'embeddings' (python-list serialized as string)
        """
        emb_path = self._check_and_create_path(
            self._files_manager.embeddings_current,
            must_exist=False,
            allowed_extensions=[".npy"],
        )
        sent_path = self._files_manager.sentences_current

        core_df = load_dataframe(self._files_manager.core_dataset)
        if self.sentence_col not in core_df.columns:
            raise KeyError(f"core_dataset missing column '{self.sentence_col}'")

        sentences = clean_sentences_df(core_df, self.sentence_col, drop_empty=True, drop_duplicates=True)[
            self.sentence_col]

        if len(sentences) == 0:
            raise ValueError("No sentences found in core_dataset after cleaning.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            LOGGER.info("Encoding with multi-process pool across %d GPUs with batch size %d", num_gpus, self.export_core_embeddings_batch_size)
            pool = self._sbert_instance.start_multi_process_pool()
            try:
                embeddings = self._sbert_instance.encode_multi_process(
                    sentences.tolist(),
                    pool,
                    batch_size=self.export_core_embeddings_batch_size,
                    show_progress_bar=True,
                ).astype(np.float32, copy=False)
            finally:
                self._sbert_instance.stop_multi_process_pool(pool)
        else:
            LOGGER.info("Encoding with single-process pool across %d GPUs with batch size %d", num_gpus, self.export_core_embeddings_batch_size)
            embeddings = self._sbert_instance.encode(
                sentences.tolist(),
                batch_size=self.export_core_embeddings_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            ).astype(np.float32, copy=False)

        emb_path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Saving embeddings -> %s", emb_path)
        np.save(emb_path, embeddings.astype(np.float32, copy=False))

        LOGGER.info("Saving sentences -> %s", sent_path)
        np.save(sent_path, np.array(sentences.tolist(), dtype=object))

        LOGGER.info("Done exporting embeddings and sentences.")

        # Compute embedding drift against previous iteration
        self._embedding_drift = self._compute_embedding_drift()

    def _compute_embedding_drift(self) -> Optional[float]:
        """Compare current vs previous embeddings (first 10k) using memory-mapped .npy files.

        After drift is computed the previous embeddings and sentences .npy files
        are deleted to save disk space.
        """
        cur_npy = self._files_manager.embeddings_current
        prev_npy = self._files_manager.embeddings_previous
        if prev_npy is None:
            LOGGER.info("No previous embeddings path; skipping drift calculation.")
            return None

        prev_sent = self._files_manager.sentences_previous

        if not cur_npy.exists() or not prev_npy.exists():
            LOGGER.info("Missing .npy file(s) for drift (cur=%s, prev=%s); skipping.",
                        cur_npy.exists(), prev_npy.exists())
            return None

        LOGGER.info("Computing embedding drift: %s vs %s", cur_npy.name, prev_npy.name)

        # Memory-map both files read-only; only the sliced rows are paged in
        cur_mmap = np.load(cur_npy, mmap_mode='r')
        prev_mmap = np.load(prev_npy, mmap_mode='r')

        n = min(100_000, cur_mmap.shape[0], prev_mmap.shape[0])
        cur = cur_mmap[:n].astype(np.float32, copy=True)
        prev = prev_mmap[:n].astype(np.float32, copy=True)

        # Release the memory maps
        del cur_mmap, prev_mmap

        # Cosine distance = 1 - cosine_similarity
        cur_norm = np.linalg.norm(cur, axis=1, keepdims=True)
        prev_norm = np.linalg.norm(prev, axis=1, keepdims=True)
        cur_norm = np.maximum(cur_norm, 1e-10)
        prev_norm = np.maximum(prev_norm, 1e-10)
        cos_sim = np.sum((cur / cur_norm) * (prev / prev_norm), axis=1)
        cos_dist = 1.0 - cos_sim
        avg_drift = float(np.mean(cos_dist))

        LOGGER.info("Average cosine distance shift (n=%d): %.6f", n, avg_drift)

        # Clean up .npy files from 3 iterations ago to allow safe resume
        cleanup_iter = self._files_manager.current_iteration - 3
        if cleanup_iter >= 0:
            old_embeddings = self._files_manager.embeddings_iteration(cleanup_iter)
            old_sentences = self._files_manager.sentences_iteration(cleanup_iter)
            for old_path in (old_embeddings, old_sentences):
                if old_path is not None and old_path.exists():
                    try:
                        old_path.unlink()
                        LOGGER.info("Removed previous file: %s", old_path)
                    except Exception as e:
                        LOGGER.warning("Failed to remove %s: %s", old_path, e)

        return avg_drift

    def _evaluate_pairs(self, selected_pairs_path: Path) -> None:
        """Evaluates the pool pairs, calculates cosine, and saves the results.
        Returns the pool DataFrame with cosine scores."""
        _selected_pairs_path = self._check_and_create_path(selected_pairs_path, must_exist=True)
        LOGGER.info('Evaluating pool pairs from %s', _selected_pairs_path)
        output_path = self._check_and_create_path(self._files_manager.evaluated_pairs_current, must_exist=False)
        if output_path.exists():
            LOGGER.error(f'Output file {output_path} already exists.')
            raise FileExistsError(f'Output file {output_path} already exists.')

        selected_pairs_dataset = self._get_dataset(
            _selected_pairs_path,
            use_random_state=False,
            score_required=False,
        )

        cosine_df = calc_cosine(self._sbert_instance, selected_pairs_dataset, **{'show_progress_bar': False})

        # Saving the result.
        original_df = load_dataframe(_selected_pairs_path)
        original_df.drop(columns=['cosine', 'cosine_norm', 'bin'], inplace=True, errors='ignore')
        core_dataset_cosine_df = original_df.merge(cosine_df, on='ID', how='left')
        save_dataframe_single(core_dataset_cosine_df, output_path, exists_ok=True)
        LOGGER.info('Done saving pairs cosine to %s', output_path)

    @staticmethod
    def _parse_cli(argv: Optional[List[str]]) -> tuple[Dict[str, Any], Optional[str]]:
        if argv is None:
            return {}, None
        p = ArgumentParser(
            description="BWS Four-Tuple Generator",
            allow_abbrev=False,
        )

        # Config / keys
        p.add_argument("--config", type=str, help="Path to config YAML")
        p.add_argument("--keys", dest="keys_path", type=str, help="Path to keys YAML (API keys)")

        # Model config
        p.add_argument("--hf-base-model", dest="hf_base_model", type=str, help="HuggingFace base model name or path")

        # Fit config
        p.add_argument("--num-train-epochs", dest="num_train_epochs", type=int, help="Number of training epochs")
        p.add_argument("--learning-rate", dest="learning_rate", type=float, help="Learning rate")
        p.add_argument("--warmup_ratio", dest="warmup_ratio", type=int, help="Warmup ratio")
        p.add_argument(
            "--eval-strategy",
            dest="eval_strategy",
            type=str,
            choices=["no", "steps", "epoch"],
            help="Evaluation strategy",
        )
        p.add_argument(
            "--logging-strategy",
            dest="logging_strategy",
            type=str,
            choices=["no", "steps", "epoch"],
            help="Logging strategy",
        )
        p.add_argument(
            "--save-strategy",
            dest="save_strategy",
            type=str,
            choices=["no", "steps", "epoch", "best"],
            help="Model saving strategy",
        )
        p.add_argument("--per-device-train-batch-size",
                       dest="per_device_train_batch_size",
                       type=int, help="Per-device training batch size")
        p.add_argument("--gradient-accumulation-steps",
                       dest="gradient_accumulation_steps",
                       type=int, help="Gradient accumulation steps")
        p.add_argument(
            "--bf16",
            action="store_true",
            default=True,
            help="Whether to use bf16 mixed precision training",
        )
        p.add_argument(
            "--load-best-model-at-end",
            dest="load_best_model_at_end",
            action="store_true",
            help="Whether to load the best model at the end of training",
        )
        p.add_argument(
            "--no-load-best-model-at-end",
            dest="load_best_model_at_end",
            action="store_false",
            help="Whether to load the best model at the end of training",
        )
        p.add_argument(
            "--greater-is-better",
            dest="greater_is_better",
            action="store_true",
            default=True,
            help="Whether a greater metric value indicates a better model",
        )
        p.add_argument("--save-total-limit",
                       dest="save_total_limit",
                       type=int, help="Total number of saved checkpoints to keep")
        p.add_argument(
            "--lr-scheduler-type",
            dest="lr_scheduler_type",
            type=str,
            choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup',
                     'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'cosine_warmup_with_min_lr',
                     'warmup_stable_decay'], help='The scheduler type to use')
        p.add_argument(
            "--pooling-strategy",
            dest="pooling_strategy",
            type=str,
            choices=["mean", "max", "cls"],
            help="Pooling strategy for SentenceTransformer",
        )
        p.add_argument(
            "--supervised-loss-type",
            dest="supervised_loss_type",
            type=str,
            choices=SUPERVISED_LOSSES,
            help="Loss function to use for supervised training",
        )
        p.add_argument(
            "--unsupervised-loss-type",
            dest="unsupervised_loss_type",
            type=str,
            choices=UNSUPERVISED_LOSSES,
            help="Loss function to use for unsupervised training",
        )

        # Embeddings export
        p.add_argument("--sentence-col",
                       dest="sentence_col",
                       type=str, help="Column name for sentences in core dataset")
        p.add_argument("--embeddings-col",
                       dest="embeddings_col",
                       type=str, help="Column name for embeddings in exported embeddings file")

        # Misc
        p.add_argument("--seed", type=int, help="Random seed")

        args, _ = p.parse_known_args(argv)
        raw = vars(args)

        maybe_config_path = raw.pop("config", None)

        cli_cfg = {k: v for k, v in raw.items() if v is not None}
        return cli_cfg, maybe_config_path
