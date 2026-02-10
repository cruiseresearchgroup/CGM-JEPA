import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

import copy
import os
import wandb
import numpy as np
from multiprocessing import Pool

from tqdm import tqdm

from data_loaders.data_loader import get_jepa_loaders
from models.encoder import Encoder
from models.glucodensity_spatial_encoder import GlucodensitySpatialEncoder
from models.predictor import Predictor
from models.cross_modal_predictor import CrossModalPredictor
from utils.glucodensity_utils import (
    compute_glucodensity_patches_from_cgm,
    apply_spatial_mask
)
from utils.mask_utils import apply_mask
from utils.main_utils import init_weights, load_device, save_model, seed_everything

from config.config_pretrain import config


def loss_pred(pred, target_ema):
    """
    Compute prediction loss between predicted and target EMA embeddings.
    
    Args:
        pred: list of predicted embeddings or tensor
        target_ema: list of target EMA embeddings or tensor
    
    Returns:
        loss: mean absolute error
    """
    if isinstance(pred, list) and isinstance(target_ema, list):
        loss = 0.0
        for pred_i, target_ema_i in zip(pred, target_ema):
            loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
        loss /= len(pred)
    else:
        loss = torch.mean(torch.abs(pred - target_ema))
    return loss


def lr_lambda(current_step):
    """Learning rate schedule with warmup."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 1.0 - progress)


def _process_single_batch_item(args):
    """Helper function for parallel KDE computation."""
    cgm_patches_single, spatial_patch_size, gridsize = args
    try:
        return compute_glucodensity_patches_from_cgm(
            cgm_patches_single,
            patch_size=spatial_patch_size,
            gridsize=gridsize
        )
    except Exception as e:
        print(f"Error in KDE computation: {e}")
        raise


def create_gluco_patches_from_cgm(cgm_patches, spatial_patch_size=8, gridsize=64, num_workers=4):
    """
    Create glucose density spatial patches from CGM temporal patches.
    Uses multiprocessing to parallelize KDE computation across batch items.
    
    Args:
        cgm_patches: (B, num_patches, patch_size) - CGM patches
        spatial_patch_size: int - size of each spatial patch
        gridsize: int - size of the KDE grid
        num_workers: int - number of parallel workers
    
    Returns:
        gluco_patches: (B, num_spatial_patches, spatial_patch_size, spatial_patch_size, 3)
    """
    B, num_patches, _ = cgm_patches.shape
    device = cgm_patches.device
    
    # Convert to numpy for KDE computation
    cgm_patches_np = cgm_patches.cpu().numpy()
    
    # Prepare arguments for parallel processing
    process_args = [
        (cgm_patches_np[b:b+1], spatial_patch_size, gridsize)
        for b in range(B)
    ]
    
    # Use multiprocessing to parallelize KDE computation
    if num_workers > 1 and B > 1:
        with Pool(processes=min(num_workers, B)) as pool:
            gluco_patches_list = pool.map(_process_single_batch_item, process_args)
    else:
        # Sequential processing for small batches or single worker
        gluco_patches_list = [_process_single_batch_item(args) for args in process_args]
    
    # Stack and convert back to tensor
    gluco_patches = np.stack(gluco_patches_list, axis=0)  # (B, num_spatial_patches, ...)
    gluco_patches = torch.tensor(gluco_patches, dtype=torch.float32).to(device)
    
    return gluco_patches

if __name__ == "__main__":
    device = load_device()
    # init config and args
    seed_everything(config["random_seed"])

    # Check for pre-computed glucodensity cache
    gluco_cache_path = config.get("gluco_cache_path", None)
    if gluco_cache_path and os.path.exists(gluco_cache_path):
        print(f"✅ Using pre-computed glucodensity patches from {gluco_cache_path}")
    elif gluco_cache_path:
        print(f"⚠️  Warning: Cache path provided but not found: {gluco_cache_path}")
        print(f"   Will compute glucodensity on-the-fly")
        gluco_cache_path = None
    else:
        print(f"ℹ️  No cache path provided. Computing glucodensity on-the-fly")
        print(f"   To speed up training, set 'gluco_cache_path' in config to pre-computed cache file")

    # load data
    loader = get_jepa_loaders(
        config["path_data"],
        config["batch_size"],
        config["patch_size"],
        config["use_time_feature"],
        config["mask_ratio"],
        gluco_cache_path=gluco_cache_path
    )

    loader.dataset.compute_stats(normalize_x=True)

    input_dim = len(loader.dataset[0][0][0])
    num_cgm_patches = len(loader.dataset[0][0])
    
    # Glucose density parameters
    gluco_spatial_patch_size = config.get("gluco_spatial_patch_size", 8)
    gluco_gridsize = config.get("gluco_gridsize", 32)
    gluco_kde_workers = config.get("gluco_kde_workers", 8)
    gluco_in_channels = 3  # KDE grids: (glucose×speed), (glucose×accel), (speed×accel)
    num_patches_per_dim = gluco_gridsize // gluco_spatial_patch_size
    num_gluco_patches = num_patches_per_dim * num_patches_per_dim

    # Load separate encoders
    cgm_encoder = Encoder(
        dim_in=input_dim,
        kernel_size=config["encoder_kernel_size"],
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["encoder_embed_bias"],
        nhead=config["encoder_nhead"],
        num_layers=config["encoder_num_layers"],
        jepa=True,
        time_inp_dim=config["time_inp_dim"],
        drop_rate=config["encoder_dropout"]
    )

    gluco_encoder = GlucodensitySpatialEncoder(
        num_gluco_patches=num_gluco_patches,
        gluco_patch_size=gluco_spatial_patch_size,
        gluco_in_channels=gluco_in_channels,
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["encoder_embed_bias"],
        nhead=config["encoder_nhead"],
        num_layers=config["encoder_num_layers"],
        jepa=True,
        drop_rate=config["encoder_dropout"]
    )

    # Load predictors
    # P_cgm: Predicts masked CGM embeddings from CGM encoder context
    predictor_cgm = Predictor(
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
        time_inp_dim=config["time_inp_dim"]
    )

    # P_gluco: Cross-modal predictor - predicts masked glucose density embeddings from CGM encoder context
    predictor_gluco = CrossModalPredictor(
        num_gluco_patches=num_gluco_patches,
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"]
    )

    # Init weights
    for m in cgm_encoder.modules():
        init_weights(m)
    for m in gluco_encoder.modules():
        init_weights(m)
    for m in predictor_cgm.modules():
        init_weights(m)
    for m in predictor_gluco.modules():
        init_weights(m)

    # Optimizer includes all trainable components
    param_groups = [
        {"params": (p for n, p in cgm_encoder.named_parameters())},
        {"params": (p for n, p in gluco_encoder.named_parameters())},
        {"params": (p for n, p in predictor_cgm.named_parameters())},
        {"params": (p for n, p in predictor_gluco.named_parameters())}
    ]
    # gluco_encoder now trains (EMA tracks it as teacher)

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    cgm_encoder = cgm_encoder.to(device)
    gluco_encoder = gluco_encoder.to(device)
    predictor_cgm = predictor_cgm.to(device)
    predictor_gluco = predictor_gluco.to(device)

    # Initialize EMA encoders (both CGM and glucodensity)
    cgm_encoder_ema = copy.deepcopy(cgm_encoder)

    # Stop-gradient step in the EMA encoders
    for p in cgm_encoder_ema.parameters():
        p.requires_grad = False

    path_save = config["path_save"] + '/x_cgm_jepa.pt'

    # Initialize EMA scheduler
    ema_scheduler = iter([
        config["ema_momentum"]
        + i
        * (1 - config["ema_momentum"])
        / (config["num_epochs"] * config["ipe_scale"])
        for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
    ])

    num_batches = len(loader)
    num_epochs = config["num_epochs"]
    total_steps = num_epochs * num_batches
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_loss, total_cgm_loss, total_gluco_loss = 0.0, 0.0, 0.0
    best_loss = float("inf")

    config_metadata = {
        "num_cgm_patches": num_cgm_patches,
        "num_gluco_patches": num_gluco_patches,
        "cgm_dim_in": input_dim,
        "gluco_spatial_patch_size": gluco_spatial_patch_size,
        "gluco_gridsize": gluco_gridsize,
        "gluco_in_channels": gluco_in_channels,
        "path_data": config["path_data"],
        "batch_size": config["batch_size"],
        "patch_size": config["patch_size"],
        "use_time_feature": config["use_time_feature"],
        "mask_ratio": config["mask_ratio"],
        "encoder_kernel_size": config["encoder_kernel_size"],
        "encoder_embed_dim": config["encoder_embed_dim"],
        "encoder_embed_bias": config["encoder_embed_bias"],
        "encoder_nhead": config["encoder_nhead"],
        "encoder_num_layers": config["encoder_num_layers"],
        "time_inp_dim": config["time_inp_dim"],
        "encoder_dropout": config["encoder_dropout"],
        "predictor_embed": config["predictor_embed"],
        "predictor_nhead": config["predictor_nhead"],
        "predictor_num_layers": config["predictor_num_layers"],
        "ema_momentum": config["ema_momentum"],
        "num_epochs": config["num_epochs"],
        "ipe_scale": config["ipe_scale"],
        "warmup_ratio": config["warmup_ratio"],
        "clip_grad_max_norm": config["clip_grad_max_norm"],
        "seed": config["random_seed"],
        "lr": config["lr"],
        "gluco_loss_weight": config.get("gluco_loss_weight", 1.0)
    }

    artifact = wandb.Artifact(
        name="cgm-jepa-glucodensity-separate",
        type="model",
        metadata=config_metadata
    )

    with wandb.init(
        project="cgm-jepa-glucodensity-separate",
        notes="CGM-JEPA-GLUCO Separate Encoders Pretraining",
        config=config_metadata
    ) as run:
        # Training loop
        for epoch in range(config["num_epochs"]):
            total_loss = 0.0
            total_cgm_loss = 0.0
            total_gluco_loss = 0.0
            m = next(ema_scheduler)
            cgm_encoder.train()
            predictor_cgm.train()
            predictor_gluco.train()
            gluco_encoder.train()

            for batch_data in tqdm(loader, desc=f"Epoch {epoch}"):
                optimizer.zero_grad()

                # Unpack batch data
                if len(batch_data) == 5:
                    # Pre-computed glucodensity patches from cache
                    patches, timestamp_patches, masks, non_masks, gluco_patches = batch_data
                    gluco_patches = gluco_patches.to(device)
                else:
                    # Compute glucodensity on-the-fly
                    patches, timestamp_patches, masks, non_masks = batch_data
                    B, num_patches, patch_size = patches.shape
                    gluco_patches = create_gluco_patches_from_cgm(
                        patches,
                        spatial_patch_size=gluco_spatial_patch_size,
                        gridsize=gluco_gridsize,
                        num_workers=gluco_kde_workers
                    )
                    gluco_patches = gluco_patches.to(device)

                patches = patches.to(device)
                timestamp_patches = timestamp_patches.to(device)
                masks = masks.to(device)
                non_masks = non_masks.to(device)

                B, num_patches, patch_size = patches.shape

                # Create independent masks for glucose density using spatial masking
                gluco_patches_sample = gluco_patches[0].cpu().numpy()
                _, gluco_mask_indices, gluco_non_mask_indices = apply_spatial_mask(
                    gluco_patches_sample,
                    mask_ratio=config["mask_ratio"],
                    mask_strategy='random'
                )

                # Convert to tensors and expand to batch dimension
                gluco_masks_1d = torch.tensor(gluco_mask_indices, dtype=torch.long).to(device)
                gluco_non_masks_1d = torch.tensor(gluco_non_mask_indices, dtype=torch.long).to(device)
                gluco_masks = gluco_masks_1d.unsqueeze(0).repeat(B, 1)
                gluco_non_masks = gluco_non_masks_1d.unsqueeze(0).repeat(B, 1)

                # Predict targets using EMA CGM encoder; use trainable gluco encoder for gluco targets
                with torch.no_grad():
                    # EMA-CGM encoder: encode all CGM patches (no masking for targets)
                    target_ema_cgm, _ = cgm_encoder_ema(patches, timestamp_patches, mask=None)
                    target_ema_cgm = F.layer_norm(target_ema_cgm, (target_ema_cgm.size(-1),))
                    # Extract masked patches for prediction loss
                    target_ema_cgm = apply_mask(target_ema_cgm, [masks])  

                # Trainable gluco encoder generates targets that receive gradients
                target_gluco, _ = gluco_encoder(gluco_patches, gluco_mask=None)
                target_gluco = F.layer_norm(target_gluco, (target_gluco.size(-1),))
                target_gluco = apply_mask(target_gluco, [gluco_masks])  

                # Encode CGM patches (context only - unmasked)
                cgm_tokens, _ = cgm_encoder(patches, timestamp_patches, mask=non_masks)  

                # Predict masked CGM embeddings using P_cgm
                pred_cgm = predictor_cgm(
                    encoded_vals=cgm_tokens,
                    x_mark=timestamp_patches,
                    masks=masks,
                    non_masks=non_masks
                )  

                # Predict masked glucose density embeddings using P_gluco (cross-view)
                pred_gluco = predictor_gluco(
                    cgm_context=cgm_tokens,  # Use CGM encoder context
                    gluco_masks=gluco_masks,
                    gluco_non_masks=gluco_non_masks
                ) 

                # Compute losses
                # 1. Temporal consistency loss
                cgm_loss = loss_pred(pred_cgm, target_ema_cgm)

                # 2. Cross-view alignment loss
                gluco_loss = loss_pred(pred_gluco, target_gluco)

                # Combined loss
                gluco_loss_weight = config.get("gluco_loss_weight", 1.0)
                loss = cgm_loss + gluco_loss_weight * gluco_loss

                # Backward and optimizer step
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(cgm_encoder.parameters()) +
                    list(predictor_cgm.parameters()) +
                    list(predictor_gluco.parameters()),
                    max_norm=config["clip_grad_max_norm"]
                )
                optimizer.step()
                scheduler.step()

                # Update EMA encoders
                with torch.no_grad():
                    for param_q, param_k in zip(
                        cgm_encoder.parameters(), cgm_encoder_ema.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                total_loss += loss.item()
                total_cgm_loss += cgm_loss.item()
                total_gluco_loss += gluco_loss.item()

            total_loss = total_loss / num_batches
            total_cgm_loss = total_cgm_loss / num_batches
            total_gluco_loss = total_gluco_loss / num_batches

            print(
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} - "
                f"Total Loss: {total_loss:.4f}, CGM Loss: {total_cgm_loss:.4f}, "
                f"Gluco Loss: {total_gluco_loss:.4f}"
            )

            if config["enable_wandb"]:
                run.log({
                    "lr": f"{optimizer.param_groups[0]['lr']:.3g}",
                    "loss": total_loss,
                    "cgm_loss": total_cgm_loss,
                    "gluco_loss": total_gluco_loss
                })

            # Save model checkpoint
            if epoch % 10 == 0:
                save_model("x_cgm_jepa", cgm_encoder, path_save)

        if config["enable_wandb"]:
            artifact.add_file(local_path=path_save, name="x_cgm_jepa")
            run.log_artifact(artifact)
