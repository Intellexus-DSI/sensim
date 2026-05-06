#!/bin/bash

# Use GPU 0
#set -a
#source .venv
#set +a
#
#export CUDA_VISIBLE_DEVICES=0,2
#source "$ENVIRONMENT"

# Generic Cartesian Product Function using Loops
# Usage: cartesian_product callback_function array1_name array2_name ...
cartesian_product() {
    local callback=$1
    shift
    local array_names=("$@")
    local num_arrays=${#array_names[@]}

    # Return if no arrays provided
    if [ $num_arrays -eq 0 ]; then
        return
    fi

    # Get lengths of all arrays
    local lengths=()
    for array_name in "${array_names[@]}"; do
        local -n arr="$array_name"
        lengths+=(${#arr[@]})
    done

    # Initialize indices array (all zeros)
    local indices=()
    for ((i=0; i<num_arrays; i++)); do
        indices+=(0)
    done

    # Main loop: iterate through all combinations
    while true; do
        # Build current combination
        local combination=()
        for ((i=0; i<num_arrays; i++)); do
            local array_name="${array_names[$i]}"
            local -n current_arr="$array_name"
            local idx=${indices[$i]}
            combination+=("${current_arr[$idx]}")
        done

        # Execute callback with current combination
        $callback "${combination[@]}"

        # Inner loop: increment indices (like an odometer)
        local pos=$((num_arrays - 1))
        while [ $pos -ge 0 ]; do
            indices[$pos]=$((indices[$pos] + 1))

            # Check if we need to carry over
            if [ ${indices[$pos]} -lt ${lengths[$pos]} ]; then
                break  # No carry needed
            fi

            # Carry over: reset this position and move to next
            indices[$pos]=0
            pos=$((pos - 1))
        done

        # If we've carried past the first position, we're done
        if [ $pos -lt 0 ]; then
            break
        fi
    done
}

run_timestamp=$(date +%Y-%m-%d_%H-%M-%S)

model_mbert_tibetan="Intellexus/mbert-tibetan-continual-wylie-final"
model_mpnet_base="sentence-transformers/all-mpnet-base-v2"
model_MiniLM_L12="sentence-transformers/all-MiniLM-L12-v2"
model_LaBSE="sentence-transformers/LaBSE"
model_TiBERT="CMLI-NLP/TiBERT"

#modals=("Intellexus/IntellexusBert-2.0" "Shailu1492/tibetan-modernbert-v4-b64-consecutive-segments" "Intellexus/mbert-tibetan-continual-wylie-final" "Shailu1492/tibetan-mbert-v4-b64-consecutive-segments" "Shailu1492/tibetan-mbert-v4-anchor-anchor" "Shailu1492/tibetan-modernbert-v4-anchor-anchor")
#modals=("Intellexus/mbert-tibetan-continual-wylie-final")
modals=("Shailu1492/tibetan-modernbert-v4-b64-consecutive-segments")
sets=("A" "B" "C" "D")

learning_rate=("2e-5") # Fixed
epochs=("9")
pooling=("weightedmean")
loss_type=("cosent" "angleloss") # Fixed
loss_scale=("20.0" "10.0") # Fixed
warmup_steps=("64") # Fixed?
max_seq_length=("512")
weight_decay=("0.1") # Fixed?
batch_size=("256") # Fixed
gradient_accumulation_steps=("1") # Fixed
lr_scheduler_type=('reduce_lr_on_plateau') # Fixed
result_dir=("./results")
merge_true=("true")
merge_false=("false")
merge_train_files=false #true  # If true, also runs the Gold→Synthetic sets with --merge_train_files
use_unicode_columns=false #true  # If true, passes --use_unicode_columns to use SentenceA_unicode/SentenceB_unicode columns
no_fit=false #true  # If true, passes --no_fit to skip model fitting (eval only)
save_strategy="no" # "no" = use last epoch model; "best" = use best checkpoint (load_best_model_at_end)
run_aggregate=true #false  # If true, runs aggregate_sets after training


run_time_identifier="2026-02-27_22-20-37" # top identifier.

shared_data_dir="./data/NewDataA-D"

# Optional: pass the full path to the synthetic trainset as $1 to override the defaults
if [ -n "$1" ]; then
    shared_train_filename_synthetic=$(basename "$1")
    run_time_identifier="${shared_train_filename_synthetic%.xlsx}"
    run_time_identifier="${run_time_identifier#merged_trainset_}"
    cp "$1" "${shared_data_dir}/${shared_train_filename_synthetic}"
else
    shared_train_filename_synthetic="merged_trainset_${run_time_identifier}.xlsx"
fi

synthetic_row_count=$(python -c "import pandas as pd; print(len(pd.read_excel('${shared_data_dir}/${shared_train_filename_synthetic}')))" 2>/dev/null)
if ! [[ "$synthetic_row_count" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not read row count from '${shared_data_dir}/${shared_train_filename_synthetic}'" >&2
    exit 1
fi
shared_result_filename="llms_sets_results_${run_time_identifier}_${synthetic_row_count}_grid_search.csv"

[ "$no_fit" = true ] && shared_result_filename="llms_no_fit.csv"

# Backup synthetic train file before training
mkdir -p ./train_backups
cp "${shared_data_dir}/${shared_train_filename_synthetic}" \
   "./train_backups/${shared_train_filename_synthetic%.xlsx}_${synthetic_row_count}.xlsx"

shared_model_dir=("ckpts/sts-b/${run_time_identifier}/${run_timestamp}")

run_experimentOnSBERT() {
    # Args: merge_flag model pooling epochs lr train_mode loss_type batch_size warmup_steps max_seq_length loss_scale weight_decay gradient_accumulation_steps lr_scheduler_type set_letter
    local merge_flag="$1"
    local model="$2"
    local pooling="$3"
    local epochs="$4"
    local lr="$5"
    local train_mode="$6"
    local loss_type="$7"
    local batch_size="$8"
    local warmup_steps="$9"
    local max_seq_length="${10}"
    local loss_scale="${11}"
    local weight_decay="${12}"
    local gradient_accumulation_steps="${13}"
    local lr_scheduler_type="${14}"
    local set_letter="${15}"

    local train_gold="train_pairs_${set_letter}_shuffled_600_scored.xlsx"
    local validation="validation_pairs_${set_letter}_shuffled_150_scored.xlsx"
    local test_file="test_pairs_${set_letter}_shuffled_250_scored.xlsx"
    local train1 train2
    if [ "$train_mode" = "syn" ]; then
        train1="$shared_train_filename_synthetic"
        train2="$train_gold"
    else
        train1="$train_gold"
        train2="$shared_train_filename_synthetic"
    fi

    local merge_arg="" unicode_arg="" no_fit_arg=""
    [ "$merge_flag" = "true" ] && merge_arg="--merge_train_files"
    [ "$use_unicode_columns" = true ] && unicode_arg="--use_unicode_columns"
    [ "$no_fit" = true ] && no_fit_arg="--no_fit"
    echo "Arguments (merge=$merge_flag, set=$set_letter, mode=$train_mode, unicode=$use_unicode_columns, no_fit=$no_fit, save_strategy=$save_strategy): $model $pooling $epochs $lr $loss_type $batch_size $warmup_steps $max_seq_length $loss_scale $weight_decay $gradient_accumulation_steps $lr_scheduler_type"
    python sbert_with_pretrain.py $merge_arg $unicode_arg $no_fit_arg \
        --wrong_pairs_log_file ./results/wrong_pairs_log.csv \
        --hf_base_model "$model" \
        --pooling_strategy "$pooling" \
        --epochs "$epochs" \
        --learning_rate "$lr" \
        --data_dir "$shared_data_dir" \
        --train_dir "$shared_data_dir" \
        --train_filenames "$train1 $train2" \
        --validation_filename "$validation" \
        --test_filenames "$test_file" \
        --results_filename "$shared_result_filename" \
        --model_dir "${shared_model_dir[0]}" \
        --loss_type "$loss_type" \
        --batch_size "$batch_size" \
        --warmup_steps "$warmup_steps" \
        --max_seq_length "$max_seq_length" \
        --loss_scale "$loss_scale" \
        --weight_decay "$weight_decay" \
        --gradient_accumulation_steps "$gradient_accumulation_steps" \
        --lr_scheduler_type "$lr_scheduler_type" \
        --save_strategy "$save_strategy"
    #torchrun --master_port=29505

    [ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${run_time_identifier}_${synthetic_row_count}_grid_search"
    [ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${run_time_identifier}_${synthetic_row_count}_grid_search_epochs"
}


# 2. SBERT Runs (12 arguments: modal, pooling, epochs, lr, data, train_dir, train1, val, test1, test2, result, train2)

if [ "$merge_train_files" = true ]; then
    echo "=== Run: Gold+Synthetic merged (--merge_train_files) → Gold model dirs ==="
    train_mode=("gold")
    cartesian_product run_experimentOnSBERT merge_true modals pooling epochs learning_rate train_mode loss_type batch_size warmup_steps max_seq_length loss_scale weight_decay gradient_accumulation_steps lr_scheduler_type sets
fi

echo "=== Run: Synthetic-only train → Synthetic model dirs ==="
train_mode=("syn")
cartesian_product run_experimentOnSBERT merge_false modals pooling epochs learning_rate train_mode loss_type batch_size warmup_steps max_seq_length loss_scale weight_decay gradient_accumulation_steps lr_scheduler_type sets

echo "=== Run: Gold-only train → Gold model dirs ==="
train_mode=("gold")
cartesian_product run_experimentOnSBERT merge_false modals pooling epochs learning_rate train_mode loss_type batch_size warmup_steps max_seq_length loss_scale weight_decay gradient_accumulation_steps lr_scheduler_type sets


[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${run_time_identifier}_${synthetic_row_count}_grid_search"
[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${run_time_identifier}_${synthetic_row_count}_grid_search_epochs"

# deactivate
#source .venv/bin/deactivate
