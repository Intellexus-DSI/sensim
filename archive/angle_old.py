#from email import parser

#from datasets import load_dataset
from angle_emb import AnglE, AngleDataTokenizer

import argparse
#import collections

import shutil
import os

from huggingface_hub import login

import text_utils_old
import model_utils_old
import utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Should be done when running.
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XET_NO_PROGRESS"] = "true"

# Disable dataset progress bars
from datasets import disable_progress_bars
disable_progress_bars()


def main(args_list=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate sentence similarity')

    parser.add_argument('--data_dir', default='./data/B', help='The directory that stores the data (defaults to ./data)')
    parser.add_argument('--train_dir', default='./data/B', help='The directory that stores the train data (defaults to ./data)')
    parser.add_argument('--model_dir', default='ckpts/sts-b', help='The directory that stores the output model')
    parser.add_argument('--results_dir', default='./results/4-12-all-runs', help='The directory that stores the results (defaults to ./results)')

    parser.add_argument('--train_filename', default='train_pairs_B_shuffled_600_scored.xlsx', help='The filename containing the training data')
    parser.add_argument('--validation_filename', default='validation_pairs_B_shuffled_100_scored.xlsx', help='The filename containing the validation data')
    parser.add_argument('--test_filename', default='test_pairs_B_shuffled_300_scored.xlsx', help='The filename containing the test data')
    parser.add_argument('--test2_filename', default=None, help='Optional second test file')


    parser.add_argument('--hf_base_model', default='OMRIDRORI/mbert-tibetan-continual-wylie-final', help='The Hugging Face base model identifier name')
    parser.add_argument('--hf_token', default=None, help='The Hugging Face access token')

    parser.add_argument('--results_filename', default='B-Results.csv', help='The filename to save the results')
    parser.add_argument('--keep_previous_model_in_dir', action='store_true', help='When marked the previous model with the check points will not be deleted')
    parser.add_argument('--seed', type=int, default=42, help='the random seed (defaults to 0)')

    parser.add_argument('--no_fit', default=False, action='store_true', help='Do not do any fitting of the model')

    parser.add_argument('--pool_of_pairs_filename', help='The given file will be added a cosine column with the similarity score')
    parser.add_argument('--pool_of_pairs_cosine_filename', help='If pool_of_pairs_filename was given the result will output to this file')

    pretrained_config_group = parser.add_argument_group('Additional for pretrained_config', 'Additional for pretrained_config')
    pretrained_config_group.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained_model_path (defaults to None)')
    pretrained_config_group.add_argument('--pretrained_lora_path', type=str, default=None, help='pretrained_lora_path (defaults to None)')
    pretrained_config_group.add_argument('--is_llm', action='store_true', help='Set this flag if the model is an LLM (defaults to False)')
    pretrained_config_group.add_argument('--pooling_strategy', type=str, default='cls', help='pooling_strategy (defaults to cls)')
    pretrained_config_group.add_argument('--train_mode', type=bool, default=False, help='train_mode (defaults to False)')
    pretrained_config_group.add_argument('--local_files_only', type=bool, default=False, help='If True will load the files from local path')

    loss_kwargs_group = parser.add_argument_group('Additional for loss_kwargs', 'Additional for loss_kwargs')
    loss_kwargs_group.add_argument('--cosine_w', type=float, default=1e-2, help='cosine_w (defaults to 1e-2)')
    loss_kwargs_group.add_argument('--ibn_w', type=float, default=1.0, help='ibn_w (defaults to 1.0)')
    loss_kwargs_group.add_argument('--cln_w', type=float, default=1.0, help='cln_w (defaults to 1.0)')
    loss_kwargs_group.add_argument('--angle_w', type=float, default=0.02, help='angle_w (defaults to 0.02)')
    loss_kwargs_group.add_argument('--cosine_tau', type=float, default=20, help='cosine_tau (defaults to 20)')
    loss_kwargs_group.add_argument('--ibn_tau', type=float, default=20, help='ibn_tau (defaults to 20)')
    loss_kwargs_group.add_argument('--angle_tau', type=float, default=20, help='angle_tau (defaults to 20)')

    fit_config_group = parser.add_argument_group('Additional for fit_config', 'Additional for fit_config')
    fit_config_group.add_argument('--batch_size', type=int, default=32, help='batch_size (defaults to 32)')
    fit_config_group.add_argument('--epochs', type=int, default=5, help='epochs (defaults to 5)')
    fit_config_group.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate (defaults to 2e-5)')
    fit_config_group.add_argument('--save_steps', type=int, default=100, help='save_steps (defaults to 100)')
    fit_config_group.add_argument('--eval_steps', type=int, default=1000, help='eval_steps (defaults to 1000)')
    fit_config_group.add_argument('--warmup_steps', type=int, default=0, help='warmup_steps (defaults to 0)')
    fit_config_group.add_argument('--gradient_accumulation_steps', type=int, default=16, help='gradient_accumulation_steps (defaults to 16)')
    fit_config_group.add_argument('--fp16', action='store_true', default=True, help='fp16 (defaults to True)')
    fit_config_group.add_argument('--logging_steps', type=int, default=100, help='logging_steps (defaults to 100)')

    args = parser.parse_args(args_list)

    # Loading datasets.

    df_mapping = {'SentenceA': 'text1', 'SentenceB': 'text2', 'score': 'label'}
    datasets_config = {
        'train_filename': args.train_filename,
        'validation_filename': args.validation_filename,
        'test_filename': args.test_filename,
        'test2_filename': args.test2_filename,
        'random_state': args.seed,
    }

    # getting the datasets
    train_dataset, validation_dataset, test_dataset, test2_dataset = text_utils_old.get_datasets(datasets_config, args.data_dir, args.train_dir, df_mapping)
    print('Datasets loaded')

    # Loading the pretrained model.
    model_name_or_path = args.hf_base_model
    pretrained_model_path = args.pretrained_model_path

    pretrained_config = {
        'model_name_or_path': model_name_or_path,
        'pretrained_model_path': pretrained_model_path,
        'pretrained_lora_path': args.pretrained_lora_path,
        'is_llm': args.is_llm,
        'pooling_strategy': args.pooling_strategy,
        'train_mode': args.train_mode,
        'local_files_only': args.local_files_only,
    }

    pretrained_ext_config = {}
    if args.hf_token:
        login(token=args.hf_token)
        pretrained_ext_config["token"] = args.hf_token

    unified_pretrained_config = pretrained_config | pretrained_ext_config

    # load pretrained model
    angle = AnglE.from_pretrained(**unified_pretrained_config).cuda()
    print('Model loaded')

    # Tokenizing data.
    train_ds = train_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    valid_ds = validation_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    test_ds = test_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    test2_ds = None
    if test2_dataset is not None:
        # Only tokenize if label exists
        if 'label' in test2_dataset.column_names:
            test2_ds = test2_dataset.map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
        else:
            print("⚠ Test2 has no labels — skipping tokenization")
            test2_ds = test2_dataset

    print('Datasets tokenized')

    # Training the model.

    # Delete checkpoint folder if it exists
    if (not args.keep_previous_model_in_dir) and os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
        print('Deleted previous model!')
    else:
        print('Previous model was kept. (if such exists)')

    loss_kwargs = {
        'cosine_w': args.cosine_w,
        'ibn_w': args.ibn_w,
        'cln_w': args.cln_w,
        'angle_w': args.angle_w,
        'cosine_tau': args.cosine_tau,
        'ibn_tau': args.ibn_tau,
        'angle_tau': args.angle_tau
    }

    fit_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'warmup_steps': args.warmup_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'fp16': args.fp16,
        'logging_steps': args.logging_steps,

    }

    # argument_kwargs = {
    #     #'device_map': 'auto',
    #     'low_cpu_mem_usage': True
    # }

    # fitting
    if not args.no_fit:
        angle.fit(
            train_ds=train_ds,
            valid_ds=valid_ds,
            output_dir=args.model_dir,
            loss_kwargs=loss_kwargs,
            #argument_kwargs=argument_kwargs,
            **fit_config
        )
        print('done fitting')

    # Evaluating.
    print('evaluating')

    train_metrics = model_utils_old.bi_encoder_evaluate(angle, train_ds, batch_size=4)
    print('Train metrics:', train_metrics)
    print('Train set pearson:', train_metrics['pearson_cosine'])

    validation_metrics = model_utils_old.bi_encoder_evaluate(angle, valid_ds, batch_size=4)
    print('Validation metrics:', validation_metrics)
    print('Validation set pearson:', validation_metrics['pearson_cosine'])

    test_metrics = model_utils_old.bi_encoder_evaluate(angle, test_ds, batch_size=4)
    print('Test metrics:', test_metrics)
    print('Test set pearson:', test_metrics['pearson_cosine'])
    test2_metrics = None
    if test2_ds is not None:
        print("Evaluating SECOND test set...")
        test2_metrics = model_utils_old.bi_encoder_evaluate(angle, test2_ds, batch_size=4)
        print("Test2 metrics:", test2_metrics)
    print('Done evaluating')

    # Logging.
    eval_result = model_utils_old.create_eval_result(
        train_metrics,
        validation_metrics,
        test_metrics,    # test1
        test2_metrics    # test2 (optional)
    )
    run_settings = {"model": "AnglE", "manual": "1", "no_fit": args.no_fit}
    run_settings.update(datasets_config)
    run_settings.update(pretrained_config)
    run_settings.update(loss_kwargs)
    run_settings.update(fit_config)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        print(f"Results folder created:{args.results_dir}")

    log_results_file = os.path.join(args.results_dir, args.results_filename)
    utils.log_evaluation_results(log_file=log_results_file, settings=run_settings, results=eval_result)

    # Evaluating pairs Cosine values to candidate_pairs
    if args.pool_of_pairs_filename:
        model_utils_old.calculate_pairs_cosine(angle, os.path.join(args.results_dir, args.pool_of_pairs_filename), os.path.join(args.results_dir, args.pool_of_pairs_cosine_filename or args.pool_of_pairs_filename))


if __name__ == '__main__':
    main()