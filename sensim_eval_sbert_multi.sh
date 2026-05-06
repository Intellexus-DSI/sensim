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

model_mbert_tibetan="Intellexus/mbert-tibetan-continual-wylie-final"
model_mpnet_base="sentence-transformers/all-mpnet-base-v2"
model_MiniLM_L12="sentence-transformers/all-MiniLM-L12-v2"
model_LaBSE="sentence-transformers/LaBSE"
model_TiBERT="CMLI-NLP/TiBERT"

#modals=("sentence-transformers/all-mpnet-base-v2")

#modals=("Intellexus/mbert-tibetan-continual-wylie-final" "Intellexus/IntellexusBert-2.0" "Shailu1492/tibetan-modernbert-v2-b64-anchor-anchor" "Shailu1492/tibetan-modernbert-v2-anchor-anchor" "Shailu1492/tibetan-mbert-v1-anchor-positive" "Shailu1492/tibetan-modernbert-anchor-positive")
modals=("Shailu1492/tibetan-modernbert-v4-b64-consecutive-segments" )
#modals=("Intellexus/mbert-tibetan-continual-wylie-final")
#modals=("sentence-transformers/LaBSE" "CMLI-NLP/TiBERT" "hfl/cino-large-v2", "Pagewood/Tibetan-BERT-wwm")
#modals=("Shailu1492/tibetan-modernbert-v4-b64-consecutive-segments" "Intellexus/IntellexusBert-2.0" "Shailu1492/tibetan-modernbert-anchor-positive" "Shailu1492/tibetan-mbert-v4-b64-consecutive-segments" "Intellexus/mbert-tibetan-continual-wylie-final" "Shailu1492/tibetan-mbert-v1-anchor-positive")

#modals=("Shailu1492/tibetan-modernbert-anchor-positive")

sets=("A" "B" "C" "D")
learning_rate=("2e-5")
epochs=("7")
pooling=("weightedmean")
loss_type=("cosent")
loss_scale=("20.0")
warmup_steps=("64")
weight_decay=("0.1")
batch_size=("256")
gradient_accumulation_steps=("1")
lr_scheduler_type=('reduce_lr_on_plateau')
max_seq_length=("512")
result_dir=("./results")
merge_true=("true")
merge_false=("false")
merge_train_files=false #true  # If true, also runs the Gold+Synthetic merged run
use_unicode_columns=false #true  # If true, passes --use_unicode_columns to use SentenceA_unicode/SentenceB_unicode columns
no_fit=false #true  # If true, passes --no_fit to skip model fitting (eval only)
save_strategy="no" # "no" = use last epoch model; "best" = use best checkpoint (load_best_model_at_end)
run_aggregate=true #false  # If true, runs aggregate_sets after training


#run_time_identifiers=("2026-02-27_12-24-40") # 5,000 mined, minimum_faiss 60, local_models
#run_time_identifiers=("2026-02-27_22-20-37") # 5,000 mined, minimum_faiss 60, BWS (Claude + Gemini) * 2
run_time_identifiers=("2026-02-27_22-20-37" "2026-02-27_12-24-40") # Total 10,000 mined, minimum_faiss 60, BWS (Claude + Gemini) * 2 + local_models

run_timestamp=$(date +%Y-%m-%d_%H-%M-%S)

shared_data_dir="./data/NewDataA-D"

# Optional: pass the full path to the synthetic trainset as $1 to override the defaults
if [ -n "$1" ]; then
    shared_train_filename=$(basename "$1")
    _id="${shared_train_filename%.xlsx}"
    _id="${_id#merged_trainset_}"
    run_time_identifiers=("$_id")
    cp "$1" "${shared_data_dir}/${shared_train_filename}"
fi

# Build synthetic file list and combined identifier
shared_train_filenames_synthetic=()
total_synthetic_row_count=0
for _id in "${run_time_identifiers[@]}"; do
    _fname="merged_trainset_${_id}.xlsx"
    shared_train_filenames_synthetic+=("$_fname")
    _count=$(python -c "import pandas as pd; print(len(pd.read_excel('${shared_data_dir}/${_fname}')))" 2>/dev/null)
    if ! [[ "$_count" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not read row count from '${shared_data_dir}/${_fname}'" >&2
        exit 1
    fi
    total_synthetic_row_count=$(( total_synthetic_row_count + _count ))
done

combined_identifier=""
for _i in "${!run_time_identifiers[@]}"; do
    _letter=$(printf "\\$(printf '%03o' $((65 + _i)))")  # A, B, C, ...
    [ -n "$combined_identifier" ] && combined_identifier+="-"
    combined_identifier+="${_letter}${run_time_identifiers[$_i]}"
done
run_time_identifier="$combined_identifier"  # kept for backward-compat references

shared_result_filename="llms_sets_results_${combined_identifier}_${total_synthetic_row_count}.csv"
shared_model_dir_synthetic="ckpts/sts-b/synthetic"
shared_model_dir=("ckpts/sts-b/${combined_identifier}/${run_timestamp}")

[ "$no_fit" = true ] && shared_result_filename="llms_no_fit.csv"

# Backup all synthetic train files before training
mkdir -p ./train_backups
for _fname in "${shared_train_filenames_synthetic[@]}"; do
    cp "${shared_data_dir}/${_fname}" \
       "./train_backups/${_fname%.xlsx}_${total_synthetic_row_count}.xlsx"
done

run_experimentOnSBERT() {
    # Args: merge_flag model pooling epochs lr train_mode loss_type batch_size warmup_steps max_seq_length loss_scale set_letter
    local merge_flag="$1"
    local model="$2"
    local pooling="$3"
    local epochs="$4"
    local lr="$5"
    local train_mode="$6"   # "syn" = synthetic-first, "gold" = gold-first
    local loss_type="$7"
    local batch_size="$8"
    local warmup_steps="$9"
    local max_seq_length="${10}"
    local loss_scale="${11}"
    local set_letter="${12}"

    local train_gold="train_pairs_${set_letter}_shuffled_600_scored.xlsx"
    local validation="validation_pairs_${set_letter}_shuffled_150_scored.xlsx"
    local test_file="test_pairs_${set_letter}_shuffled_250_scored.xlsx"
    local train_files
    if [ "$train_mode" = "syn" ]; then
        train_files="${shared_train_filenames_synthetic[*]} $train_gold"
    else
        train_files="$train_gold ${shared_train_filenames_synthetic[*]}"
    fi

    local merge_arg="" unicode_arg="" no_fit_arg=""
    [ "$merge_flag" = "true" ] && merge_arg="--merge_train_files"
    [ "$use_unicode_columns" = true ] && unicode_arg="--use_unicode_columns"
    [ "$no_fit" = true ] && no_fit_arg="--no_fit"
    echo "Arguments (merge=$merge_flag, set=$set_letter, mode=$train_mode, unicode=$use_unicode_columns, no_fit=$no_fit, save_strategy=$save_strategy): $model $pooling $epochs $lr $loss_type $batch_size $warmup_steps $max_seq_length $loss_scale"
    python sbert_with_pretrain.py $merge_arg $unicode_arg $no_fit_arg \
        --wrong_pairs_log_file ./results/wrong_pairs_log.csv \
        --hf_base_model "$model" \
        --pooling_strategy "$pooling" \
        --epochs "$epochs" \
        --learning_rate "$lr" \
        --data_dir "$shared_data_dir" \
        --train_dir "$shared_data_dir" \
        --train_filenames "$train_files" \
        --validation_filename "$validation" \
        --test_filenames "$test_file" \
        --results_filename "$shared_result_filename" \
        --model_dir "${shared_model_dir[0]}" \
        --loss_type "$loss_type" \
        --batch_size "$batch_size" \
        --warmup_steps "$warmup_steps" \
        --max_seq_length "$max_seq_length" \
        --loss_scale "$loss_scale" \
        --gradient_accumulation_steps "$gradient_accumulation_steps" \
        --save_strategy "$save_strategy"
    #torchrun --master_port=29505

    [ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${combined_identifier}_${total_synthetic_row_count}"
}


# 2. SBERT Runs (12 arguments: modal, pooling, epochs, lr, data, train_dir, train1, val, test1, test2, result, train2)

if [ "$merge_train_files" = true ]; then
    echo "=== Run: Gold+Synthetic merged (--merge_train_files) → Gold model dirs ==="
    train_mode=("gold")
    cartesian_product run_experimentOnSBERT merge_true modals pooling epochs learning_rate train_mode loss_type batch_size warmup_steps max_seq_length loss_scale sets
fi

echo "=== Run: Synthetic-only train → Synthetic model dirs ==="
train_mode=("syn")
cartesian_product run_experimentOnSBERT merge_false modals pooling epochs learning_rate train_mode loss_type batch_size warmup_steps max_seq_length loss_scale sets

echo "=== Run: Gold-only train → Gold model dirs ==="
train_mode=("gold")
cartesian_product run_experimentOnSBERT merge_false modals pooling epochs learning_rate train_mode loss_type batch_size warmup_steps max_seq_length loss_scale sets


[ "$run_aggregate" = true ] && python -m sub_tasks.aggregate_sets --run_time_identifier "${combined_identifier}_${total_synthetic_row_count}"

# deactivate
#source .venv/bin/deactivate