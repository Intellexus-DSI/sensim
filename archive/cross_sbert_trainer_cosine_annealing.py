import argparse
import os
import shutil
from datasets import disable_progress_bars
disable_progress_bars()

import numpy as np
from scipy.stats import pearsonr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import text_utils
import model_utils_pretrain
import utils_for_pretrain


# ============================================================
# METRICS FUNCTION (pearson for similarity)
# ============================================================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    return {"pearson": pearsonr(preds, labels)[0]}


# ============================================================
# TOKENIZATION
# ============================================================
def tokenize_fn(tokenizer):
    def tok(batch):
        return tokenizer(
            batch["text1"],
            batch["text2"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
    return tok


def prepare_dataset(raw_ds, tokenizer):
    ds = raw_ds.map(tokenize_fn(tokenizer), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "labels"])
    return ds


# ============================================================
# TRAINER BUILDER
# ============================================================
def build_trainer(model, args, train_ds, valid_ds, tokenizer):
    print("\n⚙️  Building Trainer with cosine annealing + best model loading...")

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        overwrite_output_dir=True,

        # Core training settings
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,

        # Logging
        logging_steps=args.logging_steps,

        # Scheduler
        lr_scheduler_type="cosine_with_restarts",

        # Save + evaluation strategy (must match!)
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,

        # Best model loading
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,

        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("✅ Trainer built successfully.\n")
    return trainer


# ============================================================
# MAIN SCRIPT
# ============================================================
def main(args_list=None):
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir')
    parser.add_argument('--train_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--results_dir')

    # Files
    parser.add_argument('--train_filename')
    parser.add_argument('--train2_filename')
    parser.add_argument('--validation_filename')
    parser.add_argument('--test_filename')
    parser.add_argument('--test2_filename')

    # Model + hyperparameters
    parser.add_argument('--hf_base_model')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=300)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--results_filename')
    parser.add_argument('--keep_previous_model_in_dir', action="store_true")

    args = parser.parse_args(args_list)

    # Mapping from Excel → dataset columns
    df_mapping = {'SentenceA': 'text1', 'SentenceB': 'text2', 'score': 'label'}

    # Load datasets
    train_ds, train2_ds, valid_ds, test_ds, test2_ds = text_utils.get_datasets(
        {
            'train_filename': args.train_filename,
            'train2_filename': args.train2_filename,
            'validation_filename': args.validation_filename,
            'test_filename': args.test_filename,
            'test2_filename': args.test2_filename,
            'random_state': 42,
        },
        args.data_dir,
        args.train_dir,
        df_mapping
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_base_model)

    # Encode datasets
    train_ds_enc = prepare_dataset(train_ds, tokenizer)
    valid_ds_enc = prepare_dataset(valid_ds, tokenizer)

    # Remove previous checkpoint folder
    if os.path.exists(args.model_dir) and not args.keep_previous_model_in_dir:
        print(f"🗑️  Removing previous model directory: {args.model_dir}")
        shutil.rmtree(args.model_dir)

    # =====================================================
    # 🔵 STAGE 1: FIRST TRAINING
    # =====================================================
    print("\n========================")
    print("🔵 STAGE 1 TRAINING START")
    print("========================")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_base_model,
        num_labels=1,
        problem_type="regression",
    )

    trainer = build_trainer(model, args, train_ds_enc, valid_ds_enc, tokenizer)
    trainer.train()

    print("\n🔍 Trainer claims best model loaded into trainer.model")
    best_model = trainer.model

    # Evaluate Stage 1
    print("\n📊 Evaluating Stage 1 best model...")
    train_metrics = model_utils_pretrain.cross_encoder_evaluate(best_model, train_ds)
    validation_metrics = model_utils_pretrain.cross_encoder_evaluate(best_model, valid_ds)
    test_metrics = model_utils_pretrain.cross_encoder_evaluate(best_model, test_ds)
    test2_metrics = model_utils_pretrain.cross_encoder_evaluate(best_model, test2_ds) if test2_ds else None

    results_file = os.path.join(args.results_dir, args.results_filename)
    utils_for_pretrain.log_evaluation_results(
        log_file=results_file,
        settings={"stage": "first"},
        results=model_utils_pretrain.create_eval_result(
            train_metrics, validation_metrics, test_metrics, test2_metrics
        )
    )

    print("✅ Stage 1 results saved.\n")

    # =====================================================
    # 🔵 STAGE 2: SECOND TRAINING (if provided)
    # =====================================================
    if args.train2_filename:
        print("\n========================")
        print("🔵 STAGE 2 TRAINING START")
        print("========================")

        train2_ds_enc = prepare_dataset(train2_ds, tokenizer)

        trainer = build_trainer(best_model, args, train2_ds_enc, valid_ds_enc, tokenizer)
        trainer.train()

        print("\n🔍 Trainer loaded best stage-2 model into trainer.model")
        best_model_stage2 = trainer.model

        # Evaluate Stage 2
        print("\n📊 Evaluating Stage 2 best model...")
        train_metrics2 = model_utils_pretrain.cross_encoder_evaluate(best_model_stage2, train_ds)
        validation_metrics2 = model_utils_pretrain.cross_encoder_evaluate(best_model_stage2, valid_ds)
        test_metrics2 = model_utils_pretrain.cross_encoder_evaluate(best_model_stage2, test_ds)
        test2_metrics2 = model_utils_pretrain.cross_encoder_evaluate(best_model_stage2, test2_ds) if test2_ds else None

        utils_for_pretrain.log_evaluation_results(
            log_file=results_file,
            settings={"stage": "second"},
            results=model_utils_pretrain.create_eval_result(
                train_metrics2, validation_metrics2, test_metrics2, test2_metrics2
            )
        )

        print("✅ Stage 2 results saved.\n")


if __name__ == "__main__":
    main()
