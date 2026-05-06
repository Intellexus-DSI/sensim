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

modalsCross=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" "cross-encoder/ms-marco-MiniLM-L6-v2" "CMLI-NLP/TiBERT")
modals=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L12-v2" "sentence-transformers/LaBSE" "CMLI-NLP/TiBERT")
learning_rate=("0.0002" "0.00002")
epochs=("3" "5" "7")
pooling=("cls" "cls_avg" "mean")
cosine_w=("0.01" "0.001")
angle_w=("0.01" "0.02")


result_dir=("./results/Synthetic-Gold-4-12")

data_dirA=("./data/NewDataA-D")
train_dirA=("./data/NewDataA-D")
train_filenameA_Synthetic=("llms_pairs_A_shuffled_2500_scored.xlsx")
train_filenameA_Gold=("train_pairs_A_shuffled_600_scored.xlsx")
validation_filenameA=("validation_pairs_A_shuffled_150_scored.xlsx")
test_filenameA=("test_pairs_A_shuffled_250_scored.xlsx")
test2_filenameA=("test_pairs_A_shuffled_no_positives_scored.xlsx")
result_filenameA=("A-Synthetic-Gold-AnglE-SBERT.csv")
result_filenameA_Cross=("A-Synthetic-Gold-Cross.csv")
result_filenameA_CrossGoldFirst=("A-Gold-Synthetic-Cross.csv")
model_dirA=("ckpts/sts-b/A-Synthetic-Gold-cross")

data_dirB=("./data/NewDataA-D")
train_dirB=("./data/NewDataA-D")
train_filenameB_Synthetic=("llms_pairs_B_shuffled_2500_scored.xlsx")
train_filenameB_Gold=("train_pairs_B_shuffled_600_scored.xlsx")
validation_filenameB=("validation_pairs_B_shuffled_150_scored.xlsx")
test_filenameB=("test_pairs_B_shuffled_250_scored.xlsx")
test2_filenameB=("test_pairs_B_shuffled_no_positives_scored.xlsx")
result_filenameB=("B-Synthetic-Gold-AnglE-SBERT.csv")
result_filenameB_Cross=("B-Synthetic-Gold-Cross.csv")
result_filenameB_CrossGoldFirst=("B-Gold-Synthetic-Cross.csv")
model_dirB=("ckpts/sts-b/B-Synthetic-Gold-cross")


data_dirC=("./data/NewDataA-D")
train_dirC=("./data/NewDataA-D")
train_filenameC_Synthetic=("llms_pairs_C_shuffled_2500_scored.xlsx")
train_filenameC_Gold=("train_pairs_C_shuffled_600_scored.xlsx")
validation_filenameC=("validation_pairs_C_shuffled_150_scored.xlsx")
test_filenameC=("test_pairs_C_shuffled_250_scored.xlsx")
test2_filenameC=("test_pairs_C_shuffled_no_positives_scored.xlsx")
result_filenameC=("C-Synthetic-Gold-AnglE-SBERT.csv")
result_filenameC_Cross=("C-Synthetic-Gold-Cross.csv")
result_filenameC_CrossGoldFirst=("C-Gold-Synthetic-Cross.csv")
model_dirC=("ckpts/sts-b/C-Synthetic-Gold-cross")

data_dirD=("./data/NewDataA-D")
train_dirD=("./data/NewDataA-D")
train_filenameD_Synthetic=("llms_pairs_D_shuffled_2500_scored.xlsx")
train_filenameD_Gold=("train_pairs_D_shuffled_600_scored.xlsx")
validation_filenameD=("validation_pairs_D_shuffled_150_scored.xlsx")
test_filenameD=("test_pairs_D_shuffled_250_scored.xlsx")
test2_filenameD=("test_pairs_D_shuffled_no_positives_scored.xlsx")
result_filenameD=("D-Synthetic-Gold-AnglE-SBERT.csv")
result_filenameD_Cross=("D-Synthetic-Gold-Cross.csv")
result_filenameD_CrossGoldFirst=("D-Gold-Synthetic-Cross.csv")
model_dirD=("ckpts/sts-b/D-Synthetic-Gold-cross")

run_experimentAOnAnglE() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}"
    python angle.py --hf_base_model "$1" --pooling "$2" --epochs "$3" --learning_rate "$4" --cosine_w "$5" --data_dir "$6" --train_dir "$7" --train_filename "$8" --validation_filename "$9" --test_filename "${10}"  --test2_filename "${11}" --results_filename "${12}" --train2_filename "${13}" --model_dir "${14}"
}

run_experimentAOnSBERT() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}"
    python sbert.py --hf_base_model "$1" --pooling "$2" --epochs "$3" --learning_rate "$4" --data_dir "$5" --train_dir "$6" --train_filename "$7" --validation_filename "$8" --test_filename "$9" --test2_filename "${10}" --results_filename "${11}" --train2_filename "${12}" --model_dir "${13}"
}

run_experimentAOnCrossEncoder() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}"
    python cross_sbert.py --hf_base_model "$1" --epochs "$2" --learning_rate "$3" --data_dir "$4" --train_dir "$5" --train_filename "$6" --validation_filename "$7" --test_filename "$8" --test2_filename "$9" --results_filename "${10}" --train2_filename "${11}" --model_dir "${12}"
}
# With Fit
## 1. AnglE Runs (13 arguments: modal, pooling, epochs, lr, cos_w, data, train_dir, train1, val, test1, test2, result, train2)
#cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirA train_dirA train_filenameA_Synthetic validation_filenameA test_filenameA test2_filenameA result_filenameA train_filenameA_Gold model_dirA
#cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirB train_dirB train_filenameB_Synthetic validation_filenameB test_filenameB test2_filenameB result_filenameB train_filenameB_Gold model_dirB
#cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirC train_dirC train_filenameC_Synthetic validation_filenameC test_filenameC test2_filenameC result_filenameC train_filenameC_Gold model_dirC
#cartesian_product run_experimentAOnAnglE modals pooling epochs learning_rate cosine_w data_dirD train_dirD train_filenameD_Synthetic validation_filenameD test_filenameD test2_filenameD result_filenameD train_filenameD_Gold model_dirD
#
## 2. SBERT Runs (12 arguments: modal, pooling, epochs, lr, data, train_dir, train1, val, test1, test2, result, train2)
#cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirA train_dirA train_filenameA_Synthetic validation_filenameA test_filenameA test2_filenameA result_filenameA train_filenameA_Gold model_dirA
#cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirB train_dirB train_filenameB_Synthetic validation_filenameB test_filenameB test2_filenameB result_filenameB train_filenameB_Gold model_dirB
#cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirC train_dirC train_filenameC_Synthetic validation_filenameC test_filenameC test2_filenameC result_filenameC train_filenameC_Gold model_dirC
#cartesian_product run_experimentAOnSBERT modals pooling epochs learning_rate data_dirD train_dirD train_filenameD_Synthetic validation_filenameD test_filenameD test2_filenameD result_filenameD train_filenameD_Gold model_dirD

# Synthetic + Gold
## 3. Cross-Encoder Runs (11 arguments: modal, epochs, lr, data, train_dir, train1, val, test1, test2, result, train2)
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirA train_dirA train_filenameA_Synthetic validation_filenameA test_filenameA test2_filenameA result_filenameA_Cross train_filenameA_Gold model_dirA
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirB train_dirB train_filenameB_Synthetic validation_filenameB test_filenameB test2_filenameB result_filenameB_Cross train_filenameB_Gold model_dirB
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirC train_dirC train_filenameC_Synthetic validation_filenameC test_filenameC test2_filenameC result_filenameC_Cross train_filenameC_Gold model_dirC
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirD train_dirD train_filenameD_Synthetic validation_filenameD test_filenameD test2_filenameD result_filenameD_Cross train_filenameD_Gold model_dirD

# Gold + Synthetic
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirA train_dirA train_filenameA_Gold validation_filenameA test_filenameA test2_filenameA result_filenameA_CrossGoldFirst train_filenameA_Synthetic model_dirA
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirB train_dirB train_filenameB_Gold validation_filenameB test_filenameB test2_filenameB result_filenameB_CrossGoldFirst train_filenameB_Synthetic model_dirB
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirC train_dirC train_filenameC_Gold validation_filenameC test_filenameC test2_filenameC result_filenameC_CrossGoldFirst train_filenameC_Synthetic model_dirC
cartesian_product run_experimentAOnCrossEncoder modalsCross epochs learning_rate data_dirD train_dirD train_filenameD_Gold validation_filenameD test_filenameD test2_filenameD result_filenameD_CrossGoldFirst train_filenameD_Synthetic model_dirD


# deactivate
#source .venv/bin/deactivate