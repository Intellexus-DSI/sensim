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
#modalsCross=( "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" "cross-encoder/ms-marco-MiniLM-L6-v2" "CMLI-NLP/TiBERT")
modals=("sentence-transformers/all-MiniLM-L12-v2")
modalsCross=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" "cross-encoder/ms-marco-MiniLM-L6-v2" "CMLI-NLP/TiBERT")
#modals=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L12-v2" "sentence-transformers/LaBSE" "CMLI-NLP/TiBERT")
learning_rate=("0.0002" "0.00002")
epochs=("3" "5" "7")
#pooling=("cls" "cls_avg" "mean")
pooling=("cls")
cosine_w=("0.01" "0.001")
angle_w=("0.01" "0.02")

data_train_dir=("./data/NewDataA-D")

train_filenameA=("train_pairs_A_shuffled_600_scored.xlsx")
validation_filenameA=("validation_pairs_A_shuffled_150_scored.xlsx")
test_filenameA=("test_pairs_A_shuffled_250_scored.xlsx")
test2_filenameA=("test_pairs_A_shuffled_no_positives_scored.xlsx")
result_filenameA=("A-Baseline-Angle-SBERT.csv")
result_filenameACross=("A-Baseline-Cross.csv")
model_dirA=("ckpts/sts-b/A-base")

train_filenameB=("train_pairs_B_shuffled_600_scored.xlsx")
validation_filenameB=("validation_pairs_B_shuffled_150_scored.xlsx")
test_filenameB=("test_pairs_B_shuffled_250_scored.xlsx")
test2_filenameB=("test_pairs_B_shuffled_no_positives_scored.xlsx")
result_filenameB=("B-Baseline-Angle-SBERT.csv")
result_filenameBCross=("B-Baseline-Cross.csv")
model_dirB=("ckpts/sts-b/B-base")

train_filenameC=("train_pairs_C_shuffled_600_scored.xlsx")
validation_filenameC=("validation_pairs_C_shuffled_150_scored.xlsx")
test_filenameC=("test_pairs_C_shuffled_250_scored.xlsx")
test2_filenameC=("test_pairs_C_shuffled_no_positives_scored.xlsx")
result_filenameC=("C-Baseline-Angle-SBERT.csv")
result_filenameCCross=("C-Baseline-Cross.csv")
model_dirC=("ckpts/sts-b/C-base")


train_filenameD=("train_pairs_D_shuffled_600_scored.xlsx")
validation_filenameD=("validation_pairs_D_shuffled_150_scored.xlsx")
test_filenameD=("test_pairs_D_shuffled_250_scored.xlsx")
test2_filenameD=("test_pairs_D_shuffled_no_positives_scored.xlsx")
result_filenameD=("D-Baseline-Angle-SBERT.csv")
result_filenameDCross=("D-Baseline-Cross.csv")
model_dirD=("ckpts/sts-b/D-base")

# running exp.
run_experimentAOnAnglENoFit() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9"
    python angle_old.py --no_fit --hf_base_model "$1" --pooling "$2" --data_dir "$3" --train_dir "$4" --train_filename "$5" --validation_filename "$6" --test_filename "$7" --test2_filename "$8" --results_filename "$9"
}
run_experimentAOnSBERTNoFit() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9"
    python sbert_old.py --no_fit --hf_base_model "$1" --pooling "$2" --data_dir "$3" --train_dir "$4" --train_filename "$5" --validation_filename "$6" --test_filename "$7" --test2_filename "$8" --results_filename "$9"
}

run_experimentAOnCrossEncoderNoFit() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8"
    python cross_sbert_old.py --no_fit --hf_base_model "$1" --data_dir "$2" --train_dir "$3" --train_filename "$4" --validation_filename "$5" --test_filename "$6" --test2_filename "$7" --results_filename "$8"
}

##
#cartesian_product run_experimentAOnCrossEncoderNoFit modalsCross data_train_dir data_train_dir train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameACross
#cartesian_product run_experimentAOnCrossEncoderNoFit modalsCross data_train_dir data_train_dir train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameBCross
#cartesian_product run_experimentAOnCrossEncoderNoFit modalsCross data_train_dir data_train_dir train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameCCross
#cartesian_product run_experimentAOnCrossEncoderNoFit modalsCross data_train_dir data_train_dir train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameDCross

#
## No Fit
#cartesian_product run_experimentAOnAnglENoFit modals pooling data_train_dir data_train_dir train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameA
#cartesian_product run_experimentAOnSBERTNoFit modals pooling data_train_dir data_train_dir train_filenameA validation_filenameA test_filenameA test2_filenameA result_filenameA
#
cartesian_product run_experimentAOnAnglENoFit modals pooling data_train_dir data_train_dir train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameB
#cartesian_product run_experimentAOnSBERTNoFit modals pooling data_train_dir data_train_dir train_filenameB validation_filenameB test_filenameB test2_filenameB result_filenameB
#
#cartesian_product run_experimentAOnAnglENoFit modals pooling data_train_dir data_train_dir train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameC
#cartesian_product run_experimentAOnSBERTNoFit modals pooling data_train_dir data_train_dir train_filenameC validation_filenameC test_filenameC test2_filenameC result_filenameC

#cartesian_product run_experimentAOnAnglENoFit modals pooling data_train_dir data_train_dir train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameD
#cartesian_product run_experimentAOnSBERTNoFit modals pooling data_train_dir data_train_dir train_filenameD validation_filenameD test_filenameD test2_filenameD result_filenameD


# deactivate
#source .venv/bin/deactivate