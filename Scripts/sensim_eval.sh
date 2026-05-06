#!/bin/bash

# Use GPU 2
export CUDA_VISIBLE_DEVICES=0

source ~/.venv/bin/activate

model_mbert_tibetan="--hf_base_model OMRIDRORI/mbert-tibetan-continual-wylie-final"
model_mpnet_base="--hf_base_model sentence-transformers/all-mpnet-base-v2"

no_fit="--no_fit"
cosine_w_0="--cosine_w 0"
cosine_w_1e_1="--cosine_w 1e-1"
cosine_w_1e_2="--cosine_w 1e-2"
cosine_w_1e_3="--cosine_w 1e-3"
cosine_w_1e_4="--cosine_w 1e-4"
cosine_w_1e_5="--cosine_w 1e-5"

configs=(
    "${model_mbert_tibetan}"
    "${model_mpnet_base}"
    "${model_mbert_tibetan} ${no_fit}"
    "${model_mpnet_base} ${no_fit}"
    "${model_mbert_tibetan} ${cosine_w_0}"
    "${model_mpnet_base} ${cosine_w_0}"
    "${model_mbert_tibetan} ${cosine_w_1}"
    "${model_mpnet_base} ${cosine_w_1}"
    "${model_mbert_tibetan} ${cosine_w_2}"
    "${model_mpnet_base} ${cosine_w_2}"
    "${model_mbert_tibetan} ${cosine_w_3}"
    "${model_mpnet_base} ${cosine_w_3}"
    "${model_mbert_tibetan} ${cosine_w_4}"
    "${model_mpnet_base} ${cosine_w_4}"
    "${model_mbert_tibetan} ${cosine_w_5}"
    "${model_mpnet_base} ${cosine_w_5}"
)

FAILED_CONFIGS=()
SUCCESSFUL_CONFIGS=()

for i in "${!configs[@]}"; do
    config="${configs[$i]}"

    echo "========================================"
    echo "Running configuration $((i+1))/${#configs[@]}: $config"
    echo "========================================"

    if python angle.py $config; then
        echo "✓ Configuration $((i+1)) completed successfully!"
        SUCCESSFUL_CONFIGS+=("$((i+1))")
    else
        echo "✗ Configuration $((i+1)) failed!"
        FAILED_CONFIGS+=("$((i+1))")
    fi

done

echo "========================================"
echo "Summary:"
echo "Successful: ${#SUCCESSFUL_CONFIGS[@]}/${#configs[@]}"
echo "Failed: ${#FAILED_CONFIGS[@]}/${#configs[@]}"

if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo "Failed configurations: ${FAILED_CONFIGS[*]}"
    exit 1
fi

# deactivate
source ~/.venv/bin/deactivate