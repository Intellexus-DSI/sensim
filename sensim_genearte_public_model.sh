#!/bin/bash

# Use GPU 0
#set -a
#source .venv
#set +a
#
#export CUDA_VISIBLE_DEVICES=0,1
#source "$ENVIRONMENT"

run_timestamp=$(date +%Y-%m-%d_%H-%M-%S)

shared_data_dir="./data/NewDataA-D"
shared_model_dir=("models/ckpts/public/${run_timestamp}")
final_model_dir="models/ckpts/public/${run_timestamp}/final"
shared_result_filename="llms_cross_sets_results/${run_timestamp}.csv"

model_mbert_tibetan="OMRIDRORI/mbert-tibetan-continual-wylie-final"
hf_repo_id="Intellexus/Cross-Dharma-Tibetan-EWTS"

## Intellexus/Cross-Dharma-Tibetan-EWTS for DH2026.
#
#python cross_sbert_with_pretrain.py \
#    --hf_base_model "$model_mbert_tibetan" \
#    --epochs "3" \
#    --learning_rate "3e-05" \
#    --data_dir "$shared_data_dir" \
#    --train_dir "$shared_data_dir" \
#    --train_filenames "merged_trainset_2026-02-27_22-20-37.xlsx all_gold_pairs_1000_scored.xlsx" \
#    --validation_filename "all_gold_pairs_1000_scored.xlsx" \
#    --test_filenames "all_gold_pairs_1000_scored.xlsx" \
#    --results_filename "$shared_result_filename" \
#    --model_dir "${shared_model_dir[0]}" \
#    --final_model_dir "$final_model_dir" \
#    --warmup_steps "100" \
#    --batch_size "64" \
#    --save_strategy "epoch" \
#    --keep_previous_model_in_dir
#
#python sub_tasks/model_tasks.py \
#    --folder "$final_model_dir" \
#    --repo-id "$hf_repo_id" \
#    --commit-message "Public Tibetan model in EWTS trained on Synthetic -> Gold data" \
#    --keys "./keys.yaml" \
#    --license "apache-2.0"

## Intellexus/Bi-Dharma-Tibetan-EWTS for DH2026.
## ── Bi-Encoder (sbert_with_pretrain.py) ─────────────────────────────────────
#
 sbert_model_dir="models/ckpts/public/${run_timestamp}"
 sbert_final_model_dir="models/ckpts/public/${run_timestamp}/final"
 sbert_result_filename="llms_sbert_results/${run_timestamp}.csv"
 sbert_hf_repo_id="Intellexus/Bi-Dharma-Tibetan-EWTS"

 python sbert_with_pretrain.py \
     --hf_base_model "$model_mbert_tibetan" \
     --epochs "7" \
     --learning_rate "2e-05" \
     --data_dir "$shared_data_dir" \
     --train_dir "$shared_data_dir" \
     --train_filenames "merged_trainset_2026-02-27_22-20-37.xlsx all_gold_pairs_1000_scored.xlsx" \
     --validation_filename "all_gold_pairs_1000_scored.xlsx" \
     --test_filenames "all_gold_pairs_1000_scored.xlsx" \
     --results_filename "$sbert_result_filename" \
     --model_dir "$sbert_model_dir" \
     --final_model_dir "$sbert_final_model_dir" \
     --warmup_steps "64" \
     --batch_size "256" \
     --save_strategy "epoch" \
     --loss_type "cosent" \
     --keep_previous_model_in_dir

 python sub_tasks/model_tasks.py \
     --folder "$sbert_final_model_dir" \
     --repo-id "$sbert_hf_repo_id" \
     --commit-message "Public Tibetan bi-encoder in EWTS trained on Synthetic -> Gold data" \
     --keys "./keys.yaml" \
     --license "apache-2.0"




# deactivate
#source .venv/bin/deactivate
