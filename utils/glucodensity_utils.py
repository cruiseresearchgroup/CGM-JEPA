import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde
import torch


def compute_2d_kde_grid(a, b, gridsize=64):
    """
    Compute 2D KDE density grid (image-like) for a pair of variables.
    
    Args:
        a: numpy array - first variable (e.g., glucose)
        b: numpy array - second variable (e.g., speed)
        gridsize: int - size of the grid (gridsize x gridsize)
    
    Returns:
        Z: numpy array of shape (gridsize, gridsize) - KDE density grid
        a_grid: numpy array - grid values for variable a
        b_grid: numpy array - grid values for variable b
    """
    # Create KDE from data points
    X = np.vstack([a, b])
    kde = gaussian_kde(X, bw_method="scott")
    
    # Get percentile ranges (1st to 99th) to avoid outliers
    a_min, a_max = np.percentile(a, [1, 99])
    b_min, b_max = np.percentile(b, [1, 99])
    
    # Create grid
    a_grid = np.linspace(a_min, a_max, gridsize)
    b_grid = np.linspace(b_min, b_max, gridsize)
    Ag, Bg = np.meshgrid(a_grid, b_grid)
    
    # Evaluate KDE on grid points
    grid_points = np.vstack([Ag.ravel(), Bg.ravel()])
    Z = kde(grid_points).reshape(Ag.shape)
    
    # Normalize for stability
    Z = Z / (Z.max() + 1e-12)
    
    return Z, a_grid, b_grid


def compute_glucodensity_grids(cgm_sequence, smoothing_factor=1.0, gridsize=64):
    """
    Compute glucodensity features as 2D KDE grids (image-like).
    Creates three 2D grids for pairs:
    - (glucose, speed)
    - (glucose, acceleration)
    - (speed, acceleration)
    
    Args:
        cgm_sequence: numpy array of shape (T,) - CGM values for one day (288 points)
        smoothing_factor: smoothing factor for spline interpolation
        gridsize: int - size of each KDE grid (gridsize x gridsize)
    
    Returns:
        grids: numpy array of shape (gridsize, gridsize, 3) - stacked KDE grids
               Channel 0: glucose × speed
               Channel 1: glucose × acceleration
               Channel 2: speed × acceleration
    """
    if isinstance(cgm_sequence, torch.Tensor):
        cgm_sequence = cgm_sequence.cpu().numpy()
    
    # Ensure 1D
    if cgm_sequence.ndim > 1:
        cgm_sequence = cgm_sequence.flatten()
    
    n_points = len(cgm_sequence)
    
    # Time axis: 1 day, 5-min interval
    t_hours = (np.arange(n_points) * 5.0) / 60.0  # 0, 0.0833, ..., 23.9167
    
    # Smooth + derivatives via spline
    spline = interpolate.UnivariateSpline(t_hours, cgm_sequence, s=smoothing_factor)
    
    G_smooth = spline(t_hours)
    dG_dt = spline.derivative(1)(t_hours)       # speed (mg/dL per hour)
    ddG_dt2 = spline.derivative(2)(t_hours)     # acceleration (mg/dL per hour²)
    
    # Compute 2D KDE grids for each pair
    # 1) Glucose × Speed
    Z_glu_speed, _, _ = compute_2d_kde_grid(G_smooth, dG_dt, gridsize=gridsize)
    
    # 2) Glucose × Acceleration
    Z_glu_accel, _, _ = compute_2d_kde_grid(G_smooth, ddG_dt2, gridsize=gridsize)
    
    # 3) Speed × Acceleration
    Z_speed_accel, _, _ = compute_2d_kde_grid(dG_dt, ddG_dt2, gridsize=gridsize)
    
    # Stack grids as channels (H, W, C) format - image-like
    grids = np.stack([
        Z_glu_speed,
        Z_glu_accel,
        Z_speed_accel
    ], axis=-1)  # (gridsize, gridsize, 3)
    
    return grids


def create_spatial_patches_from_grid(grid, patch_size=8):
    """
    Create spatial patches from 2D KDE grid (Vision Transformer-style patching).
    
    Args:
        grid: numpy array of shape (H, W, C) - 2D KDE grid image
        patch_size: int - size of each spatial patch (patch_size x patch_size)
    
    Returns:
        patches: numpy array of shape (num_patches, patch_size, patch_size, C)
                 where num_patches = (H // patch_size) * (W // patch_size)
    """
    H, W, C = grid.shape
    
    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Truncate grid to fit patches exactly
    H_trunc = num_patches_h * patch_size
    W_trunc = num_patches_w * patch_size
    grid_trunc = grid[:H_trunc, :W_trunc, :]
    
    # Reshape into patches: (num_patches_h, patch_size, num_patches_w, patch_size, C)
    # then transpose to (num_patches_h, num_patches_w, patch_size, patch_size, C)
    patches = grid_trunc.reshape(
        num_patches_h, patch_size,
        num_patches_w, patch_size,
        C
    ).transpose(0, 2, 1, 3, 4)  # (num_patches_h, num_patches_w, patch_size, patch_size, C)
    
    # Flatten spatial dimensions: (num_patches, patch_size, patch_size, C)
    num_patches = num_patches_h * num_patches_w
    patches = patches.reshape(num_patches, patch_size, patch_size, C)
    
    return patches


def apply_spatial_mask(patches, mask_ratio=0.25, mask_strategy='random'):
    """
    Apply spatial masking to glucodensity patches (image-style masking).
    
    Args:
        patches: numpy array of shape (num_patches, patch_size, patch_size, C)
        mask_ratio: float - ratio of patches to mask
        mask_strategy: str - 'random' for random masking, 'block' for block masking
    
    Returns:
        masked_patches: numpy array - patches with masked regions set to 0
        mask_indices: numpy array - indices of masked patches
        non_mask_indices: numpy array - indices of non-masked patches
    """
    num_patches = patches.shape[0]
    num_masked = int(num_patches * mask_ratio)
    
    if mask_strategy == 'random':
        # Random patch masking (like MAE)
        mask_indices = np.random.choice(num_patches, size=num_masked, replace=False)
    elif mask_strategy == 'block':
        # Block masking (contiguous regions)
        start_idx = np.random.randint(0, num_patches - num_masked + 1)
        mask_indices = np.arange(start_idx, start_idx + num_masked)
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
    
    non_mask_indices = np.setdiff1d(np.arange(num_patches), mask_indices)
    
    # Create masked version (set masked patches to zero)
    masked_patches = patches.copy()
    masked_patches[mask_indices] = 0.0
    
    return masked_patches, mask_indices, non_mask_indices


def compute_glucodensity_patches_from_cgm(cgm_patches, patch_size=8, gridsize=32, smoothing_factor=1.0):
    """
    Compute glucodensity spatial patches from CGM patches.
    Creates 2D KDE grids and extracts spatial patches (Vision Transformer-style).
    
    Args:
        cgm_patches: numpy array of shape (num_patches, patch_size) - CGM patches
        patch_size: int - size of each spatial patch (patch_size x patch_size)
        gridsize: int - size of the KDE grid (gridsize x gridsize)
        smoothing_factor: float - smoothing factor for spline interpolation
    
    Returns:
        glu_patches: numpy array of shape (num_patches, patch_size, patch_size, 3)
                     Spatial patches from stacked KDE grids
    """
    # Flatten patches to get full sequence
    cgm_sequence = cgm_patches.flatten()
    
    # Compute 2D KDE grids (image-like)
    grids = compute_glucodensity_grids(
        cgm_sequence, 
        smoothing_factor=smoothing_factor,
        gridsize=gridsize
    )  # (gridsize, gridsize, 3)
    
    # Create spatial patches from grid
    glu_patches = create_spatial_patches_from_grid(grids, patch_size=patch_size)
    # (num_spatial_patches, patch_size, patch_size, 3)
    
    return glu_patches

