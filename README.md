# Pediatric Pneumonia Detection System

A complete deep learning pipeline for binary classification of chest X-rays.

## Project Structure

- `model_core/`: Core modules for data pipeline, model building, training, and evaluation.
- `notebooks/`: Jupyter notebooks for experimentation.
- `run/`: Shell scripts for running the pipeline.
- `scripts/`: Python scripts for training and evaluation.

## Usage

### Training

To train the model:

```bash
python scripts/train.py --dataset_path /path/to/chest_xray --stage1_epochs 8 --stage2_epochs 5
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --model_path /path/to/model.h5 --dataset_path /path/to/chest_xray
```

## Requirements

See `requirements.txt`.
