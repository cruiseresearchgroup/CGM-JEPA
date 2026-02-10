import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

import copy
import os
import json
import wandb

from tqdm import tqdm

from data_loaders.data_loader import get_jepa_loaders
from models.encoder import Encoder
from models.predictor import Predictor

from utils.mask_utils import apply_mask
from utils.main_utils import init_weights, load_device, seed_everything, save_model

from config.config_pretrain import config

def loss_pred(pred, target_ema):
    loss = 0.0
    for pred_i, target_ema_i in zip(pred, target_ema):
        loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
    loss /= len(pred)
    return loss

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 1.0 - progress)

if __name__ == "__main__":

    # Init preparation
    device = load_device()
    seed_everything(config["random_seed"])

    # load data
    loader = get_jepa_loaders(
        config["path_data"],
        config["batch_size"],
        config["patch_size"],
        config["use_time_feature"],
        config["mask_ratio"]
    )

    loader.dataset.compute_stats(normalize_x=True)

    input_dim = len(loader.dataset[0][0][0])

    # load encoder
    encoder = Encoder(
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

    # load predictor 
    predictor = Predictor(
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
        time_inp_dim=config["time_inp_dim"]
    )

    for m in encoder.modules():
        init_weights(m)
    
    for m in predictor.modules():
        init_weights(m)

    param_groups = [
        {"params": (p for n, p in encoder.named_parameters())},
        {"params": (p for n, p in predictor.named_parameters())}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])
    encoder = encoder.to(device)
    predictor = predictor.to(device)

    # initialize the EMA-encoder
    encoder_ema = copy.deepcopy(encoder)

    # stop-gradient step in the EMA
    for p in encoder_ema.parameters():
        p.requires_grad = False

    path_save = config["path_save"] + '/cgm_jepa.pt'

    # initialize the EMA Scheduler
    ema_scheduler = (
        config["ema_momentum"]
        + i
        * (1 - config["ema_momentum"])
        / (config["num_epochs"] * config["ipe_scale"])
        for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
    )

    num_batches = len(loader)
    num_epochs = config["num_epochs"]
    total_steps = num_epochs * num_batches
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda
    )

    total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0

    config_metadata = {
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
            "lr": config["lr"]
    }

    artifact = wandb.Artifact(
        name="cgm-jepa", 
        type="model",
        metadata=config_metadata
    )

    with wandb.init(
        project="cgm-jepa", 
        notes="CGM-JEPA Pretraining",
        config=config_metadata
    ) as run:
        # training loop
        for epoch in range(config["num_epochs"]):
            total_loss = 0.0
            m = next(ema_scheduler)
            encoder.train()
            predictor.train()

            for patches, timestamp_patches, masks, non_masks in tqdm(loader, desc=f"Epoch {epoch}"):
                optimizer.zero_grad()

                patches = patches.to(device)
                timestamp_patches = timestamp_patches.to(device)
                masks = masks.to(device)
                non_masks = non_masks.to(device)

                # predict targets
                with torch.no_grad():
                    target_ema, _ = encoder_ema(patches, timestamp_patches)        
                    target_ema = F.layer_norm(
                        target_ema, (target_ema.size(-1),)
                    )
                    # takes the representation for the masked patches for prediction loss
                    target_ema = apply_mask(target_ema, masks)                  
                
                # encode and predict the masked tokens
                tokens, _ = encoder(patches, timestamp_patches, mask=non_masks)    
                pred = predictor(tokens, x_mark=timestamp_patches, masks=masks, non_masks=non_masks)

                # compute the loss
                loss = loss_pred(pred, target_ema)

                # backward and optimizer step
                loss.backward()

                nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), max_norm=config["clip_grad_max_norm"])
                
                optimizer.step()
                scheduler.step()

                # update the EMA
                with torch.no_grad():
                    for param_q, param_k in zip(
                        encoder.parameters(), encoder_ema.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
                    
                total_loss += loss

            total_loss = total_loss / num_batches

            print(
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} - JEPA Loss: {total_loss:.4f}"
            )

            if config["enable_wandb"]:
                run.log({
                    "lr": f"{optimizer.param_groups[0]['lr']:.3g}",
                    "loss": total_loss
                })

            # save model's checkpoint
            if epoch % 10 == 0 and epoch != 0:
                save_model("cgm_jepa", encoder, path_save)
        
        if config["enable_wandb"]:
            artifact.add_file(local_path=path_save, name="cgm-jepa")
            run.log_artifact(artifact)