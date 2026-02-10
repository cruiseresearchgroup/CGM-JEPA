"""
Shared TS2Vec utilities for integration with class_reg and model_configs.

Provides TS2VecEncoderWrapper, load_pretrained_ts2vec, and freeze_ts2vec
so that model_configs can add TS2Vec to the evaluation pipeline without
circular imports.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

from models.ts2vec.ts2vec import *

class TS2VecEncoderWrapper(nn.Module):
    """
    Wrapper to make TS2Vec compatible with the Trainer system.
    TS2Vec expects (n_samples, n_timestamps, n_features); this adapts flat input.
    """
    def __init__(self, ts2vec_model: TS2Vec, device: str):
        super().__init__()
        self.ts2vec_model = ts2vec_model
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 2:
            x_np = x_np[:, :, np.newaxis]
        x_np[x_np == -1] = np.nan
        with torch.no_grad():
            embeddings = self.ts2vec_model.encode(
                x_np, encoding_window='full_series', batch_size=batch_size
            )
        embeddings = torch.from_numpy(embeddings).float().to(self.device)
        return embeddings, embeddings


def freeze_ts2vec(ts2vec_model: TS2Vec) -> TS2Vec:
    """Freeze all parameters of the TS2Vec encoder."""
    for attr in ("_net", "net"):
        if hasattr(ts2vec_model, attr):
            module = getattr(ts2vec_model, attr)
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad = False
            if hasattr(module, "eval"):
                module.eval()
    return ts2vec_model


def load_pretrained_ts2vec(
    checkpoint_path: str,
    device: str,
    input_dims: int = 1,
    output_dims: int = 96,
    hidden_dims: int = 64,
    depth: int = 10,
) -> TS2Vec:
    """Load a pretrained TS2Vec from checkpoint and freeze it."""
    model = TS2Vec(
        input_dims=input_dims,
        output_dims=output_dims,
        hidden_dims=hidden_dims,
        depth=depth,
        device=device,
        lr=0.001,
        batch_size=16,
    )
    model.load(checkpoint_path)
    freeze_ts2vec(model)
    return model
