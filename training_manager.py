import os
import shutil

from datetime import datetime

import argparse

from sympy.strategies.branch import debug

import utils
import angle
import bws_processing
import llms

import app_config


def exe_progressive_learner(args_list=None):
    parser = argparse.ArgumentParser(description='Run the iterative progressive learner')
    parser.add_argument('--pool_of_pairs_filename', default='candidate_pairs.xlsx',
                        help='The filename containing the pairs to iterative progressive learn')
    parser.add_argument('--data_dir', default='./data', help='The directory that stores the data (defaults to ./data)')
    parser.add_argument('--results_dir', default='./results/llms',
                        help='The directory that will store the results (defaults to ./results)')
    parser.add_argument('--bws_dir', default='./bws', help='The working directory for bws (defaults to ./bws)')
    parser.add_argument('--results_filename', default='progressive_learner.csv',
                        help='The csv to save the correlations')
    parser.add_argument('--k', default=5, type=int, help='Number of pairs in batch (defaults to 1000)')
    parser.add_argument('--bins', default=5, type=int, help='Number of bins (defaults to 10)')

    args = parser.parse_args(args_list)

    now = datetime.now()
    base_results_folder = os.path.join(args.results_dir, now.strftime('%Y-%m-%d_%H-%M-%S'))
    model_dir = os.path.join(base_results_folder, 'ckpts/sts-b')
    os.makedirs(base_results_folder, exist_ok=True)  # Creates llms/{date}
    shutil.copy(os.path.join(args.data_dir, args.pool_of_pairs_filename),
                os.path.join(base_results_folder,
                             args.pool_of_pairs_filename))
    # Copies candidate_pairs.xlsx from data/candidate_pairs.xlsx

    shutil.copy(app_config.CONFIG_FILE_NAME, os.path.join(base_results_folder, app_config.CONFIG_FILE_NAME))  # Copies config.yaml to llms/{date}/config.yaml

    base_angle_args = [
        '--results_dir', base_results_folder,  # args.results_dir,
        '--data_dir', args.data_dir,
    ]

    # Run the model with no fit at first as there is no training data and create a cosine for the pairs.
    pool_of_pairs_cosine_filename = f'pool_of_pairs_cosine_00.xlsx'  # initialized to 0
    angle_args = [
        '--no_fit',
        '--pool_of_pairs_filename', args.pool_of_pairs_filename,
        '--pool_of_pairs_cosine_filename', pool_of_pairs_cosine_filename,
        '--results_filename', args.results_filename,
        '--model_dir', model_dir,
        '--keep_previous_model_in_dir',
    ]
    angle_args.extend(base_angle_args)
    angle.main(angle_args)  # Creates progressive_learner.csv and pool_of_pairs_cosine_00.xlsx in llms/{date}

    for i in range(2):

        bws_dir = os.path.join(base_results_folder, f'it_{i:02d}')
        results_dir = os.path.join(base_results_folder, f'it_{i:02d}')

        os.makedirs(bws_dir, exist_ok=True)  # Creates llms/{date}/it_{i:02d}
        os.makedirs(results_dir, exist_ok=True)  # Creates llms/{date}/it_00 (redundant)

        if i == 0:
            shutil.move(os.path.join(base_results_folder, pool_of_pairs_cosine_filename),
                        os.path.join(results_dir,
                                     pool_of_pairs_cosine_filename))  # Copies pool_of_pairs_cosine_00.xlsx to llms/{date}/it_00
        else:
            previous_results_dir = os.path.join(base_results_folder, f'it_{i - 1:02d}')
            shutil.move(os.path.join(previous_results_dir, pool_of_pairs_cosine_filename),
                        os.path.join(results_dir,
                                     pool_of_pairs_cosine_filename))  # Moves pool_of_pairs_cosine_{i:02d}.xlsx from llms/{date}/it_{i-1:02d} to llms/{date}/it_{i:02d}

        pool_of_pairs_cosine_filepath = os.path.join(results_dir, f'pool_of_pairs_cosine_{i:02d}.xlsx')
        selected_pairs_filepath = os.path.join(results_dir, f'selected_pairs_{i:02d}.xlsx')
        candidate_pool_of_pairs_filename = f'candidate_pool_{i:02d}.xlsx'
        candidate_pool_of_pairs_filepath = os.path.join(results_dir, candidate_pool_of_pairs_filename)

        temp_ids_filename = f'temp_ids_{i:02d}.txt'
        sampled_4_pairs_filename = f'sampled_4_pairs_{i:02d}.xlsx'  # We have issues with this file type

        # Slice the pairs file to selected pairs and rest.
        utils.split_input_by_distribution(pool_of_pairs_cosine_filepath, selected_pairs_filepath,
                                          candidate_pool_of_pairs_filepath, args.k, args.bins,
                                          column='cosine')  # Creates selected_pairs_{i:02d}.xlsx and candidate_pool_{i:02d}.xlsx in llms/{date}/it_{i:02d}

        # Create 4 pairs tuples file for annotation.
        bws_processing.generate_bws_tuples(selected_pairs_filepath, work_dir=bws_dir,
                                           temp_ids_filename=temp_ids_filename,
                                           sampled_4_pairs_filename=sampled_4_pairs_filename)  # Creates sampled_4_pairs_{i:02d}.xlsx, temp_ids_{i:02d}.txt and temp_ids_{i:02d}.txt.tuples in llms/{date}/it_{i:02d}

        # Annotated the 4 pairs tuples file with the LLMs.
        #models=["gemini-2.5-flash", "gpt-4o-mini", "claude-3-haiku-20240307"]
        models=["gemini-2.5-flash"]
        # models = ["DEBUG"]
        for m in models:
            llms.run(input_file=sampled_4_pairs_filename,
                     data_dir=bws_dir,
                     model=m,
                     results_dir=results_dir,
                     output_file=f'similarity_results_{m}.csv',
                     # debug=True
                     )  # Creates similarity_results_{m}.csv in llms/{date}/it_{i:02d} for each model

        # Combine all LLMs results into a single file.
        llms_4_pair_annotated_files = [os.path.join(results_dir, f'similarity_results_{m}.csv') for m in models]
        llms_4_pair_annotations_filename = f'llms_4_pair_annotations_{i:02d}.xlsx'
        llms_4_pair_annotations_filepath = os.path.join(bws_dir, llms_4_pair_annotations_filename)
        utils.concatenate_csv_xls_files(llms_4_pair_annotated_files,
                                        llms_4_pair_annotations_filepath)  # Concatenates and creates llms_4_pair_annotations_{i:02d}.xlsx in llms/{date}/it_{i:02d}.xslx

        # Calculate BWS from the annotated 4 pairs tuples file.
        dest_pairs_scored_filename = f'llms_pairs_scored_{i:02d}.xlsx'
        bws_processing.calculate_bws_scores(selected_pairs_filepath, llms_4_pair_annotations_filename,
                                            dest_pairs_scored_filename,
                                            work_dir=bws_dir)  # Creates llms_pairs_scored_{i:02d}.xlsx, llms_pairs_scored_{i:02d}_scored.tsv, and llms_pairs_scored_{i:02d}_bws.formatted  in llms/{date}/it_{i:02d}

        # Train the model and save it + create a cosine for the pairs.
        pool_of_pairs_cosine_filename = f'pool_of_pairs_cosine_{i + 1:02d}.xlsx'  # Calculating the cosine for next round.
        angle_args = [
            '--train_dir', bws_dir,
            '--train_filename', dest_pairs_scored_filename,
            '--pool_of_pairs_filename', candidate_pool_of_pairs_filename,
            '--pool_of_pairs_cosine_filename', pool_of_pairs_cosine_filename,
            '--results_filename', args.results_filename,
            '--results_dir', results_dir,  # args.results_dir,
            '--data_dir', args.data_dir,
            '--model_dir', model_dir,
            '--keep_previous_model_in_dir',
        ]
        if i > 0:
            angle_args.extend(['--hf_base_model', model_dir, '--pretrained_model_path', model_dir])
        # angle_args.extend(base_angle_args)

        angle.main(angle_args)  # Creates pool_of_pairs_cosine_{i + 1:02d}.xlsx and progressive_learner.csv in llms/{date}/it_{i:02d}


if __name__ == '__main__':
    exe_progressive_learner()
