"""Tiny-Edit-Guard training with Hugging Face Hub data (data/dataloader_hf.py).

Thin wrapper around train.py with `--data-source hf` as the default, so both entry points share
the same model factory, batching (padding / pair batches / paired augmentation), loss and
defaults. All train.py options are accepted.

Example:
    python train_hf.py --model patch_topk --epochs 1 --max-train-batches 5 --max-val-batches 2
"""
from train import main

if __name__ == "__main__":
    main(data_source_default="hf")
