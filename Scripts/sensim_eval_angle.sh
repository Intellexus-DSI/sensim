#!/bin/bash

# Use GPU 0
export CUDA_VISIBLE_DEVICES=3

source .venv310/bin/activate

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
#sonar_200="cointegrated/SONAR_200_text_encoder"

modals=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L12-v2" "sentence-transformers/LaBSE")
learning_rate=("0.002" "0.0002" "0.00002")
epochs=("3" "5" "7")
pooling=("cls" "cls_avg" "mean")
cosine_w=("0.01" "0.001")
angle_w=("0.01" "0.02")

# running exp.
run_experiment() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7"

    python angle.py --no_fit --hf_base_model "$1" --pooling "$2" --epochs "$3" --learning_rate "$4" --cosine_w "$5" #--angle_w "$6"
    python angle.py --hf_base_model "$1" --pooling "$2" --epochs "$3" --learning_rate "$4" --cosine_w "$5" #--angle_w "$6"
}

# Generate all combinations
cartesian_product run_experiment modals pooling epochs learning_rate cosine_w #angle_w

# deactivate
#source .venv/bin/deactivate