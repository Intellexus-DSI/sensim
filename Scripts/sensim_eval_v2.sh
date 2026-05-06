#!/bin/bash

# Use GPU 2
export CUDA_VISIBLE_DEVICES=0

source .venv/bin/activate

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

model_mbert_tibetan="--hf_base_model OMRIDRORI/mbert-tibetan-continual-wylie-final"
model_mpnet_base="--hf_base_model sentence-transformers/all-mpnet-base-v2"
model_MiniLM_L12="--hf_base_model sentence-transformers/all-MiniLM-L12-v2"

#modals=("${model_mbert_tibetan}" "${model_mpnet_base}" "${model_MiniLM_L12}")
#fitting=("" "--no_fit")
#learning_rates=("--learning_rate 0.002" "--learning_rate 0.0002" "--learning_rate 0.00002" "--learning_rate 0.000002")
#cosine_w=("--cosine_w 0.02" "--cosine_w 0.01" "--cosine_w 0.001")
#ibn_w=("--ibn_w 1.0" "--ibn_w 0.1" "--ibn_w 0.01")
#epochs=("--epochs 3" "--epochs 5" "--epochs 7")
#pooling=("--pooling_strategy cls" "--pooling_strategy cls_avg" "--pooling_strategy mean")
#Pooling strategy [`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]

modals=("OMRIDRORI/mbert-tibetan-continual-wylie-final" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L12-v2")
#learning_rates=("0.002" "0.0002" "0.00002" "0.000002")
#epochs=("3" "5" "7")
pooling=("cls_avg" "mean" "cls" "cls_max")
cosine_w=("0.02" "0.01" "0.001")
ibn_w=("1.0" "0.1" "0.01")
#fitting=("" "--no_fit")

# --hf_base_model OMRIDRORI/mbert-tibetan-continual-wylie-final  --learning_rate 1e-5 --cosine_w 0.02 --ibn_w 1.0 --epochs 5 --pooling_strategy cls

# Your callback
run_experiment() {
    echo "Arguments: $1 $2 $3 $4 $5 $6 $7"
#    python angle.py --hf_base_model "$1" --learning_rate "$2" --epochs "$3" --pooling "$4" --cosine_w "$5" --ibn_w "$6" --no_fit --train_mode True
#    python angle.py --hf_base_model "$1" --learning_rate "$2" --epochs "$3" --pooling "$4" --cosine_w "$5" --ibn_w "$6" --train_mode True
#    python angle.py --hf_base_model "$1" --learning_rate "$2" --epochs "$3" --pooling "$4" --cosine_w "$5" --ibn_w "$6" --no_fit
#    python angle.py --hf_base_model "$1" --learning_rate "$2" --epochs "$3" --pooling "$4" --cosine_w "$5" --ibn_w "$6"


    python angle.py --hf_base_model "$1" --pooling "$2" --cosine_w "$3" --ibn_w "$4" --no_fit
    python angle.py --hf_base_model "$1" --pooling "$2" --cosine_w "$3" --ibn_w "$4"
}

# Generate all combinations
# cartesian_product run_experiment modals learning_rates epochs pooling cosine_w ibn_w
cartesian_product run_experiment modals pooling cosine_w ibn_w

# deactivate
#source .venv/bin/deactivate