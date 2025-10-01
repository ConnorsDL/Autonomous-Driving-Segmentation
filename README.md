# Autonomous Driving Semantic Segmentation

This project provides a complete training pipeline for building a semantic segmentation model that detects vehicles (or any other classes) in autonomous driving imagery. It includes dataset utilities, a UNet-based model, a configurable training loop, and example configuration files.

## Features

- **Dataset loader** expecting paired RGB images and segmentation masks stored on disk.
- **UNet architecture** implemented in PyTorch with configurable encoder widths.
- **Training utilities** with automatic mixed precision, checkpointing, and mean IoU tracking.
- **Config-driven CLI** for running experiments with reproducible settings.

## Installation

1. Create a Python virtual environment (Python 3.9+ is recommended).
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organise your dataset using the following folder structure:

```
<dataset_root>/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

- Image and mask filenames must match (e.g. `0001.png` in both folders).
- Masks should encode class indices as pixel values. If your masks use arbitrary values (e.g. 0 for background and 255 for vehicles) you can remap them via the `class_map` section of the config file.
- Optionally create a validation split with the same structure (`data/val/images`, `data/val/masks`).

## Configuration

Training is controlled via a YAML configuration file. See [`configs/example.yaml`](configs/example.yaml) for a detailed example. The main sections are:

- `data`: dataset paths, augmentation toggle, image size, and optional class remapping.
- `model`: UNet hyperparameters such as number of input channels, output classes, and encoder widths.
- `training`: optimisation settings including learning rate, number of epochs, gradient clipping, and checkpoint directory.

## Training

Run the training script by pointing it to a configuration file. Ensure the `src/` directory is on the Python path (e.g. via `PYTHONPATH=src`).

```bash
PYTHONPATH=src python train.py --config configs/example.yaml --output runs/history.json
```

Key outputs:

- Checkpoints saved to the directory defined in `training.checkpoint_dir` (best model stored as `best.pt`).
- Training history (losses and mIoUs) optionally written to the `--output` JSON file.
- Console logs summarising progress and best validation mIoU.

## Evaluating a Checkpoint

To resume training or evaluate the best checkpoint, set `training.resume_from` in the YAML config to the checkpoint path. The trainer will restore weights, optimiser state, and gradient scaler automatically.

## Project Structure

```
src/autonomous_seg/
├── __init__.py         # Package exports
├── data.py             # Dataset and dataloader utilities
├── models.py           # UNet architecture and builder
└── trainer.py          # Training loop, metrics, and checkpointing
train.py                # CLI entry point for experiments
configs/example.yaml    # Example configuration file
```

## Notes

- Mixed precision is enabled by default and will automatically fall back to full precision when training on CPU.
- Modify `training.num_classes` (inferred from the model section) and `training.ignore_index` to match your dataset annotations.
- For multi-class segmentation, ensure the masks contain all required class indices and update `model.num_classes` accordingly.
