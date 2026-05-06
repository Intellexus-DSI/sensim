#!/bin/bash

# Use GPU 0
#set -a
#source .venv
#set +a
#
#export CUDA_VISIBLE_DEVICES=0,1
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

model_mbert_tibetan="OMRIDRORI/mbert-tibetan-continual-wylie-final"
model_mpnet_base="sentence-transformers/all-mpnet-base-v2"
model_MiniLM_L12="sentence-transformers/all-MiniLM-L12-v2"
model_LaBSE="sentence-transformers/LaBSE"
model_TiBERT="CMLI-NLP/TiBERT"

#modalsCross=("Intellexus/mbert-tibetan-continual-wylie-final" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L12-v2")
#modalsCross=("sentence-transformers/LaBSE" "sangjeedondrub/tibetan-roberta-base" "CMLI-NLP/TiBERT" "hfl/cino-large-v2")
#modalsCross=("hfl/cino-large-v2" "sentence-transformers/LaBSE" "sangjeedondrub/tibetan-roberta-base")
#modalsCross=("Intellexus/mbert-tibetan-continual-wylie-final" "Shailu1492/tibetan-mbert-v1-anchor-positive" "Intellexus/IntellexusBert-2.0" "Shailu1492/tibetan-modernbert-v2-b64-anchor-anchor" )
#modalsCross=("Shailu1492/tibetan-mbert-v1-consecutive-segments")
#modalsCross=("Intellexus/IntellexusBert-2.0")
modalsCross=("Intellexus/mbert-tibetan-continual-wylie-final")
sets=("A" "B" "C" "D")
#learning_rate=("2e-5")
learning_rate=("3e-5") # 2e-5
epochs=("7")
loss_type=("binaryCrossEntropyLoss")
warmup_steps=("100") # 10
result_dir=("./results/cross_trainer")
run_aggregate=true  # If true, runs aggregate_sets after training
use_unicode_columns=false #true  # If true, passes --use_unicode_columns to use SentenceA_unicode/SentenceB_unicode columns
no_fit=false #true  # If true, passes --no_fit to skip model fitting (eval only)

run_timestamp=$(date +%Y-%m-%d_%H-%M-%S)

shared_data_dir="./data/NewDataA-D"
shared_train_filename_synthetic="merged_trainset_2026-02-27_22-20-37.xlsx"
shared_result_filename="llms_cross_sets_results_2027-05-01_13-00-0.csv"
shared_model_dir_synthetic="/mnt/temp-disk/ckpts/sts-b/synthetic" #"ckpts/sts-b/synthetic"
shared_model_dir=("/mnt/temp-disk/ckpts/sts-b/${run_timestamp}") #("ckpts/sts-b/${run_timestamp}")

[ "$no_fit" = true ] && shared_result_filename="llms_cross_sets_results_no_fit.csv"

run_experimentAOnCrossEncoder() {
    # Args: model epochs lr train_mode warmup_steps set_letter
    local model="$1"
    local epochs="$2"
    local lr="$3"
    local train_mode="$4"
    local warmup_steps="$5"
    local set_letter="$6"

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

    local unicode_arg="" no_fit_arg=""
    [ "$use_unicode_columns" = true ] && unicode_arg="--use_unicode_columns"
    [ "$no_fit" = true ] && no_fit_arg="--no_fit"

    echo "Arguments (set=$set_letter, mode=$train_mode): $model $epochs $lr $warmup_steps"
    python cross_sbert_with_pretrain.py $no_fit_arg $unicode_arg \
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
        --batch_size 64 \
        --save_strategy "best" \
        --eval_steps 100 \
        --warmup_steps "$warmup_steps"
}

## 3. Cross-Encoder Runs
train_mode=("syn")
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate train_mode warmup_steps sets

# Gold First
train_mode=("gold")
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate train_mode warmup_steps sets

echo "Results file: file://$(realpath "./results/Cross-Trainer/$shared_result_filename")"

_result_id="${shared_result_filename#llms_cross_sets_results_}"
_result_id="${_result_id%.csv}"
[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "$_result_id" --cross_trainer

# aggregate_sets already run _epochs in the above line, so no need to run it again here
#[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${_result_id}_epochs" --cross_trainer


# deactivate
#source .venv/bin/deactivate
