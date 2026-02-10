"""
Pre-compute glucodensity patches for all samples in the dataset.
This eliminates the KDE computation bottleneck during training.

Usage:
    python -m utils.precompute_glucodensity \
        --data_path /path/to/data.csv \
        --output_path /path/to/gluco_cache.pkl \
        --gluco_gridsize 32 \
        --gluco_spatial_patch_size 8
"""
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from data_loaders.data_class import JEPALoader
from utils.glucodensity_utils import compute_glucodensity_patches_from_cgm


def precompute_glucodensity_patches(
    path_data,
    output_path,
    patch_size=12,
    series_split_size=288,
    gluco_spatial_patch_size=8,
    gluco_gridsize=32,
    use_time_feature=False,
    timeenc=1,
    freq='t'
):
    """
    Pre-compute glucodensity patches for all samples in the dataset.
    
    Args:
        path_data: Path to CSV data file
        output_path: Path to save pre-computed patches (will be a pickle file)
        patch_size: Size of CGM patches
        series_split_size: Size of series splits
        gluco_spatial_patch_size: Spatial patch size for glucodensity
        gluco_gridsize: KDE grid size
        normalize: Whether to normalize data
        use_time_feature: Whether to use time features
        timeenc: Time encoding type
        freq: Frequency for time features
    
    Returns:
        Dictionary mapping (subject, split_idx) -> gluco_patches
    """
    print(f"Loading dataset from {path_data}...")
    loader = JEPALoader(
        path_data=path_data,
        is_precompute_gluco=True,
        series_split_size=series_split_size,
        patch_size=patch_size,
        mask_ratio=0.0, 
        use_time_feature=use_time_feature,
    )

    
    print(f"Found {len(loader)} samples")
    print(f"Computing glucodensity patches (gridsize={gluco_gridsize}, spatial_patch_size={gluco_spatial_patch_size})...")
    
    gluco_cache = {}
    
    for idx in tqdm(range(len(loader)), desc="Pre-computing glucodensity"):
        # Get sample (without glucodensity - we'll compute it)
        patches_tensor, _, _, _ = loader[idx]
        
        # Convert to numpy
        patches_np = patches_tensor.numpy()  # (num_patches, patch_size)
        
        # Compute glucodensity patches
        gluco_patches = compute_glucodensity_patches_from_cgm(
            patches_np,
            patch_size=gluco_spatial_patch_size,
            gridsize=gluco_gridsize
        )  # (num_spatial_patches, spatial_patch_size, spatial_patch_size, 3)
        
        # Store with sample key
        subject, split_idx = loader.samples[idx]
        gluco_cache[(subject, split_idx)] = gluco_patches
    
    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving pre-computed patches to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'gluco_patches': gluco_cache,
            'config': {
                'gluco_spatial_patch_size': gluco_spatial_patch_size,
                'gluco_gridsize': gluco_gridsize,
                'patch_size': patch_size,
                'series_split_size': series_split_size
            }
        }, f)
    
    print(f"✅ Pre-computation complete! Saved {len(gluco_cache)} samples to {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return gluco_cache


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-compute glucodensity patches")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save pre-computed patches")
    parser.add_argument("--patch_size", type=int, default=12, help="CGM patch size")
    parser.add_argument("--series_split_size", type=int, default=288, help="Series split size")
    parser.add_argument("--gluco_spatial_patch_size", type=int, default=8, help="Spatial patch size")
    parser.add_argument("--gluco_gridsize", type=int, default=32, help="KDE grid size") 
    # NOTE: We use 32 size for precomputation
    args = parser.parse_args()
    
    precompute_glucodensity_patches(
        path_data=args.data_path,
        output_path=args.output_path,
        patch_size=args.patch_size,
        series_split_size=args.series_split_size,
        gluco_spatial_patch_size=args.gluco_spatial_patch_size,
        gluco_gridsize=args.gluco_gridsize
    )
