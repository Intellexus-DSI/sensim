#!/bin/bash

# Use GPU 0
set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=3
source "$ENVIRONMENT"

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
#sonar_200="cointegrated/SONAR_200_text_encoder"
#modals=("CMLI-NLP/TiBERT")
modals=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L12-v2" "sentence-transformers/LaBSE" "CMLI-NLP/TiBERT")
learning_rate=("0.0002" "0.00002")
epochs=("3" "5" "7")
pooling=("cls" "cls_avg" "mean")
cosine_w=("0.01" "0.001")
angle_w=("0.01" "0.02")

data_dirA=("./data/NewDataA-D")
train_dirA=("./data/NewDataA-D")
train_filenameA=("train_pairs_A_shuffled_600_scored.xlsx")
validation_filenameA=("validation_pairs_A_shuffled_150_scored.xlsx")
test_filenameA=("test_pairs_A_shuffled_250_scored.xlsx")
test2_filenameA=("test_pairs_A_shuffled_no_positives_scored.xlsx")
result_filenameA=("A-600-250-150-result.csv")

data_dirB=("./data/NewDataA-D")
train_dirB=("./data/NewDataA-D")
train_filenameB=("train_pairs_B_shuffled_600_scored.xlsx")
validation_filenameB=("validation_pairs_B_shuffled_150_scored.xlsx")
test_filenameB=("test_pairs_B_shuffled_250_scored.xlsx")
test2_filenameB=("test_pairs_B_shuffled_no_positives_scored.xlsx")
result_filenameB=("B-600-250-150-result.csv")

data_dirC=("./data/NewDataA-D")
train_dirC=("./data/NewDataA-D")
train_filenameC=("train_pairs_C_shuffled_600_scored.xlsx")
validation_filenameC=("validation_pairs_C_shuffled_150_scored.xlsx")
test_filenameC=("test_pairs_C_shuffled_250_scored.xlsx")
test2_filenameC=("test_pairs_C_shuffled_no_positives_scored.xlsx")
result_filenameC=("C-600-250-150-result.csv")

data_dirD=("./data/NewDataA-D")
train_dirD=("./data/NewDataA-D")
train_filenameD=("train_pairs_D_shuffled_600_scored.xlsx")
validation_filenameD=("validation_pairs_D_shuffled_150_scored.xlsx")
test_filenameD=("test_pairs_D_shuffled_250_scored.xlsx")
test2_filenameD=("test_pairs_D_shuffled_no_positives_scored.xlsx")
result_filenameD=("D-600-250-150-result.csv")

# running exp.
run_experimentAOnAnglENoFit() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9"
    python angle.py --no_fit --hf_base_model "$1" --pooling "$2" --data_dir "$3" --train_dir "$4" --train_filename "$5" --validation_filename "$6" --test_filename "$7" --test2_filename "$8" --results_filename "$9"
}
run_experimentAOnSBERTNoFit() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9"
    python sbert.py --no_fit --hf_base_model "$1" --pooling "$2" --data_dir "$3" --train_dir "$4" --train_filename "$5" --validation_filename "$6" --test_filename "$7" --test2_filename "$8" --results_filename "$9"
}

run_experimentAOnAnglE() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}"
    python angle.py --hf_base_model "$1" --pooling "$2" --epochs "$3" --learning_rate "$4" --cosine_w "$5" --data_dir "$6" --train_dir "$7" --train_filename "$8" --validation_filename "$9" --test_filename "${10}"  --test2_filename "${11}" --results_filename "${12}"
}
run_experimentAOnSBERT() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}"
    python sbert.py --hf_base_model "$1" --pooling "$2" --epochs "$3" --learning_rate "$4" --data_dir "$5" --train_dir "$6" --train_filename "$7" --validation_filename "$8" --test_filename "$9" --test2_filename "${10}" --results_filename "${11}"
}

# Generate all combinations
# No Fit
cartesian_product run_experimentAOnAnglENoFit modals pooling data_dirA train_dirA train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameA
cartesian_product run_experimentAOnSBERTNoFit modals pooling data_dirA train_dirA train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameA

cartesian_product run_experimentAOnAnglENoFit modals pooling data_dirB train_dirB train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameB
cartesian_product run_experimentAOnSBERTNoFit modals pooling data_dirB train_dirB train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameB

cartesian_product run_experimentAOnAnglENoFit modals pooling data_dirC train_dirC train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameC
cartesian_product run_experimentAOnSBERTNoFit modals pooling data_dirC train_dirC train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameC

cartesian_product run_experimentAOnAnglENoFit modals pooling data_dirD train_dirD train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameD
cartesian_product run_experimentAOnSBERTNoFit modals pooling data_dirD train_dirD train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameD


# With Fit
cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirA train_dirA train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameA
cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirA train_dirA train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameA

cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirB train_dirB train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameB
cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirB train_dirB train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameB

cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirC train_dirC train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameC
cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirC train_dirC train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameC

cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirD train_dirD train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameD
cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirD train_dirD train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameD

# deactivate
#source .venv/bin/deactivate