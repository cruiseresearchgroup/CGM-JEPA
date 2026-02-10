import torch
import os
import json
import wandb

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn


from models.gluformer.gluformer import GluFormer
from utils.main_utils import load_device, seed_everything, save_model
from data_loaders.data_loader import get_gluformer_dataloader

from config.config_pretrain import config

from tqdm import tqdm


if __name__ == "__main__":
    
    device = load_device()
    seed_everything(config["random_seed"])

    loader = get_gluformer_dataloader(
        path=config["path_data"],
        batch_size=config["batch_size"]
    )

    vocab_size = loader.dataset.num_bins
    PAD_TOKEN = vocab_size

    artifact = wandb.Artifact(
        name="gluformer", 
        type="model",
        metadata={
            "vocab_size": vocab_size,
            "embed_dim": config["encoder_embed_dim"],
            "nhead": config["encoder_nhead"],
            "num_layers": config["encoder_num_layers"],
            "dim_feedforward": 2 * config["encoder_embed_dim"],
            "max_seq_length": 25000,
            "encoder_dropout": config["encoder_dropout"]
        }
    )

    model = GluFormer(
        vocab_size=vocab_size,
        embed_dim=config["encoder_embed_dim"],
        nhead=config["encoder_nhead"],
        num_layers=config["encoder_num_layers"],
        dim_feedforward=2 * config["encoder_embed_dim"],
        max_seq_length=25000,
        dropout=config["encoder_dropout"],
        pad_token=vocab_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    num_batches = len(loader)
    num_epochs = config["num_epochs"]
    total_steps = num_epochs * num_batches
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    best_loss = float('inf')
    path_save = config["path_save"] + '/gluformer.pt'

    with wandb.init(
        project="gluformer",
        notes="Gluformer Pretraining",
        config=config
    ) as run:
        for epoch in range(config["num_epochs"]):
            total_loss = 0.0
            model.train()
            
            for batched_tokenized_series in tqdm(loader, desc=f"Epoch {epoch}"):
                optimizer.zero_grad()

                batched_tokenized_series = batched_tokenized_series.to(device)
                mask = (batched_tokenized_series == PAD_TOKEN)

                inputs, targets = batched_tokenized_series[:, :-1], batched_tokenized_series[:, 1:]
                mask = mask[:, :-1]

                logits = model(inputs, src_key_padding_mask=mask)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=PAD_TOKEN)
                
                total_loss += loss.item()

                loss.backward()

                optimizer.step()
                scheduler.step()

            total_loss = total_loss / num_batches

            print(
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} - GluFormer Loss: {total_loss:.4f}"
            )

            if config["enable_wandb"]:
                run.log({
                    "lr": f"{optimizer.param_groups[0]['lr']:.3g}",
                    "loss": total_loss
                })

            # save model's checkpoint
            if epoch % 10 == 0 and epoch != 0:
                save_model("gluformer", model, path_save)

        if config["enable_wandb"]:
            artifact.add_file(local_path=path_save, name="gluformer")
            run.log_artifact(artifact)