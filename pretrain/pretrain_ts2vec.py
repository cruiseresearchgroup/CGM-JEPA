"""
Pretrain TS2Vec on CGM data.

This script trains TS2Vec on CGM time series data and saves the model checkpoint.
The pretrained model can then be loaded for downstream evaluation tasks.

Usage:
    python -m pretrain.pretrain_ts2vec
"""

import sys
import os
import numpy as np
import torch
import json
from copy import deepcopy
from typing import Optional
import wandb

import warnings

warnings.filterwarnings("ignore")

from models.ts2vec.ts2vec import TS2Vec

from config.config_pretrain import config
from data_loaders.data_loader import get_jepa_loaders
from utils.main_utils import load_device, seed_everything, save_model

def extract_raw_timeseries_from_loader(loader):
    """
    Extract raw time series data from a JEPA loader.
    
    JEPALoader uses CSVDataLoader which stores samples as (subject, split_idx) tuples.
    We need to access the actual time series data from the CSV file.
    
    Args:
        loader: DataLoader from get_jepa_loaders
        
    Returns:
        numpy array of shape (n_samples, n_timestamps, n_features)
    """
    all_data = []

    for batch_idx, batch in enumerate(loader):
        try:
            # JEPALoader returns (patches, timestamp_patches, mask_indices, non_mask_indices)
            # (optionally plus gluco_patches when using a cache). We only need CGM patches.
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                patches = batch[0]
            else:
                patches = batch

            # patches: (B, num_patches, patch_size)
            if patches.ndim != 3:
                print(f"Warning: Unexpected patches shape {patches.shape} in batch {batch_idx}, skipping")
                continue

            B, num_patches, patch_size = patches.shape

            # Flatten patches back into a contiguous time series per sample:
            # (B, num_patches, patch_size) -> (B, num_patches * patch_size, 1)
            series = patches.reshape(B, num_patches * patch_size, 1).cpu().numpy()

            all_data.append(series)
        except Exception as e:
            print(f"Warning: Skipping batch {batch_idx} due to error: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data samples found in loader")

    # Concatenate all batches into a single array
    data = np.concatenate(all_data, axis=0)  # (n_samples, n_timestamps, n_features)

    print(f"Extracted {data.shape[0]} samples")
    print(f"Data shape: {data.shape}")

    return data


def main():
    """Main function to pretrain TS2Vec."""
    # Parse args and get config
    device = load_device()

    seed_everything(config["random_seed"])
    
    # TS2Vec configuration
    ts2vec_config = {
        "output_dims": config.get("ts2vec_output_dims", 320),
        "hidden_dims": config.get("ts2vec_hidden_dims", 64),
        "depth": config.get("ts2vec_depth", 10),
        "lr": config.get("ts2vec_lr", 0.001),
        "batch_size": config.get("ts2vec_batch_size", 16),
        "max_train_length": config.get("ts2vec_max_train_length", 3000),
        "n_epochs": config.get("ts2vec_n_epochs", 10),
        "n_iters": config.get("ts2vec_n_iters", None),
    }
    
    # TS2Vec is known to have issues with MPS, so use CPU or CUDA
    if isinstance(device, torch.device) and device.type == "cuda":
        ts2vec_device = "cuda"
    else:
        ts2vec_device = "cpu"
        print("Note: Using CPU for TS2Vec (MPS not supported)")
    
    print("=" * 80)
    print("TS2Vec Pretraining")
    print("=" * 80)
    print(f"Patch size: {config['patch_size']}")
    print(f"Main device: {device}")
    print(f"TS2Vec device: {ts2vec_device}")
    print(f"TS2Vec config: {ts2vec_config}")
    print("=" * 80)
    
    # Load data using JEPA loader (gets raw time series)
    print("\nLoading data...")
    loader = get_jepa_loaders(
        path=config["path_data"],
        batch_size=config["batch_size"],
        patch_size=config["patch_size"],
        use_time_feature=config["use_time_feature"],
        mask_ratio=config["mask_ratio"],  # Not used for TS2Vec but needed for loader
        gluco_cache_path=None  # Not using glucodensity for TS2Vec
    )
    
    # Compute stats on the dataset (required for JEPALoader)
    print("\nComputing dataset statistics...")
    loader.dataset.compute_stats(indices=None, normalize_x=True, normalize_y=False)
    
    # Extract raw time series data
    print("\nExtracting raw time series...")
    train_data = extract_raw_timeseries_from_loader(loader)
    
    # Handle missing values: convert -1 to NaN (TS2Vec expects NaN)
    train_data[train_data == -1] = np.nan
    
    # Initialize wandb if available
    run = None
    if config.get("enable_wandb", False):
        wandb_project = config.get("wandb_project", "cgm-ts2vec-pretrain")
        run = wandb.init(
            project=wandb_project,
            name=f"ts2vec",
            config={
                **ts2vec_config,
                "patch_size": config["patch_size"],
                "n_samples": train_data.shape[0],
                "n_timestamps": train_data.shape[1],
                "n_features": train_data.shape[2],
            }
        )
    
    # Train TS2Vec
    print("\n" + "=" * 80)
    print("Training TS2Vec...")
    print("=" * 80)
    
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        output_dims=ts2vec_config["output_dims"],
        hidden_dims=ts2vec_config["hidden_dims"],
        depth=ts2vec_config["depth"],
        device=ts2vec_device,
        lr=ts2vec_config["lr"],
        batch_size=ts2vec_config["batch_size"],
        max_train_length=ts2vec_config["max_train_length"]
    )
    
    loss_log = model.fit(
        train_data,
        n_epochs=ts2vec_config["n_epochs"],
        n_iters=ts2vec_config["n_iters"],
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Final loss: {loss_log[-1] if loss_log else 'N/A'}")
    print(f"Total epochs: {model.n_epochs}")
    
    if run is not None:
        run.log({"final_loss": loss_log[-1] if loss_log else 0.0})
        run.log({"n_epochs": model.n_epochs})
    
    # Save model
    path_save = config["path_save"] + '/ts2vec.pkl'
    
    metadata = {
        "model_type": "ts2vec",
        "patch_size": config["patch_size"],
        "output_dims": ts2vec_config["output_dims"],
        "hidden_dims": ts2vec_config["hidden_dims"],
        "depth": ts2vec_config["depth"],
        "lr": ts2vec_config["lr"],
        "batch_size": ts2vec_config["batch_size"],
        "max_train_length": ts2vec_config["max_train_length"],
        "n_epochs": model.n_epochs,
        "n_samples": train_data.shape[0],
        "n_timestamps": train_data.shape[1],
        "n_features": train_data.shape[2],
        "final_loss": float(loss_log[-1]) if loss_log else None,
    }
    
    model.save(path_save)
    
    if run is not None and config.get("enable_wandb", False):
        artifact = wandb.Artifact(
            name=f"ts2vec",
            type="model",
            metadata=metadata
        )
        artifact.add_file(path_save)
        run.log_artifact(artifact)
        print(f"\nModel uploaded to wandb as artifact: ts2vec")
    
    if run is not None:
        run.finish()
    
    print("\n" + "=" * 80)
    print("TS2Vec pretraining completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
