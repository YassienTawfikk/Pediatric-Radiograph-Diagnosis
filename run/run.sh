#!/bin/bash
#SBATCH --job-name=pneumonia_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Ensure we are in the project root
cd "$(dirname "$0")/.." || exit

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting training job..."

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "Python could not be found. Please load appropriate modules or activate virtualenv."
    exit 1
fi

# Note: You can pass the dataset path as the first argument to this script
DATASET_PATH=${1:-"/path/to/dataset"}

python scripts/train.py \
  --dataset_path "$DATASET_PATH" \
  --stage1_epochs 8 \
  --stage2_epochs 5 \
  --output_dir "outputs"

echo "Job finished."
