import torch
import torch.nn as nn
import numpy as np

from ..utils.embed import DataEmbedding
from ..utils.modules import *
from ..utils.mask_utils import *

class Encoder(nn.Module):
    '''
        @brief: Encode input into latent representations. Used for
        both input and target encoder
    '''
    def __init__(
        self,
        num_patches,
        dim_in,
        kernel_size,
        embed_dim,
        embed_bias,
        nhead,
        num_layers,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        jepa=False,
        embed_activation=nn.GELU(),
        time_inp_dim=5, # depend on freq map | 't' = 5
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.activation = embed_activation if embed_activation else nn.GELU()

        self.data_embedding = DataEmbedding(
            dim=embed_dim,
            in_channels=dim_in,
            patch_size=kernel_size,
            time_inp_dim=time_inp_dim,
            dropout=drop_rate
        )

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for i in range(num_layers)
            ]
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.jepa = jepa

        self.proj = MLP(
            in_features=embed_dim,
            hidden_features=1024,
            out_features=48,
            act_layer=nn.GELU
        )

    def forward(self, x, x_mark=None, mask=None):
        # x: (B, C, T), x_mark: (B, L, time_inp_dim)
        
        # optional timestamp
        if x_mark is not None:
            x_mark = x_mark.to(x.device)

        if x_mark is None:
            x_mark = torch.zeros_like(x)

        x = self.data_embedding(x, x_mark) # (B, L, D)

        # apply mask. in the encoder, we only keep unmasked part
        if mask is not None and self.jepa:
            x = apply_mask(x, mask)

        # encode using attention
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)
        
        x = self.encoder_norm(x)

        return x, self.proj(x)