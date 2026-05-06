#!/bin/bash

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

modalsCross=("Intellexus/mbert-tibetan-continual-wylie-final")
sets=("A" "B" "C" "D")

learning_rate=("1e-5" "2e-5" "3e-5")
epochs=("10")
loss_type=("binaryCrossEntropyLoss")
warmup_steps=("100")
batch_size=("8" "16" "32" "64" "128")
weight_decay=("0.1")
gradient_accumulation_steps=("1")
lr_scheduler_type=("reduce_lr_on_plateau" "cosine_with_restarts" "cosine_with_min_lr")
save_strategy="best"
run_aggregate=true  # If true, runs aggregate_sets after training

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
shared_result_filename="llms_cross_sets_results_${run_time_identifier}_${synthetic_row_count}_grid_search.csv"

# Backup synthetic train file before training
mkdir -p ./train_backups
cp "${shared_data_dir}/${shared_train_filename_synthetic}" \
   "./train_backups/${shared_train_filename_synthetic%.xlsx}_${synthetic_row_count}.xlsx"

#shared_model_dir=("ckpts/sts-b/${run_time_identifier}/${run_timestamp}")
shared_model_dir=("/mnt/temp-disk/ckpts/sts-b/${run_timestamp}") #("ckpts/sts-b/${run_timestamp}")

run_experimentOnCrossEncoder() {
    # Args: model epochs lr train_mode loss_type batch_size warmup_steps weight_decay gradient_accumulation_steps lr_scheduler_type set_letter
    local model="$1"
    local epochs="$2"
    local lr="$3"
    local train_mode="$4"
    local loss_type="$5"
    local batch_size="$6"
    local warmup_steps="$7"
    local weight_decay="$8"
    local gradient_accumulation_steps="$9"
    local lr_scheduler_type="${10}"
    local set_letter="${11}"

    local train_gold="train_pairs_${set_letter}_shuffled_600_scored.xlsx"
    local validation="validation_pairs_${set_letter}_shuffled_150_scored.xlsx"
    local test_file="test_pairs_${set_letter}_shuffled_250_scored.xlsx"
    local test2_file="test_pairs_${set_letter}_shuffled_no_positives_scored.xlsx"
    local train1 train2
    if [ "$train_mode" = "syn" ]; then
        train1="$shared_train_filename_synthetic"
        train2="$train_gold"
    else
        train1="$train_gold"
        train2="$shared_train_filename_synthetic"
    fi

    echo "Arguments (set=$set_letter, mode=$train_mode): $model $epochs $lr $loss_type $batch_size $warmup_steps $weight_decay $gradient_accumulation_steps $lr_scheduler_type"
    python cross_sbert_with_pretrain.py \
        --hf_base_model "$model" \
        --epochs "$epochs" \
        --learning_rate "$lr" \
        --data_dir "$shared_data_dir" \
        --train_dir "$shared_data_dir" \
        --train_filenames "$train1 $train2" \
        --validation_filename "$validation" \
        --test_filenames "$test_file $test2_file" \
        --results_filename "$shared_result_filename" \
        --model_dir "${shared_model_dir[0]}" \
        --loss_type "$loss_type" \
        --batch_size "$batch_size" \
        --warmup_steps "$warmup_steps" \
        --weight_decay "$weight_decay" \
        --gradient_accumulation_steps "$gradient_accumulation_steps" \
        --lr_scheduler_type "$lr_scheduler_type" \
        --save_strategy "$save_strategy"
}

## Cross-Encoder Runs

echo "=== Run: Synthetic-first train ==="
train_mode=("syn")
cartesian_product run_experimentOnCrossEncoder modalsCross epochs learning_rate train_mode loss_type batch_size warmup_steps weight_decay gradient_accumulation_steps lr_scheduler_type sets

echo "=== Run: Gold-first train ==="
train_mode=("gold")
cartesian_product run_experimentOnCrossEncoder modalsCross epochs learning_rate train_mode loss_type batch_size warmup_steps weight_decay gradient_accumulation_steps lr_scheduler_type sets

_result_id="${run_time_identifier}_${synthetic_row_count}_grid_search"
[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "$_result_id" --cross_trainer
[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${_result_id}_epochs" --cross_trainer