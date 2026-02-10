import torch
import torch.nn as nn
import numpy as np

from utils.embed import PositionalEmbedding, TimeFeatureEmbedding
from utils.modules import *
from utils.mask_utils import apply_mask

class Predictor(nn.Module):
    '''
        @brief: Takes the representation from context encoder and predict masked
        representations
    '''
    def __init__(
        self,
        encoder_embed_dim=128,
        predictor_embed_dim=128,
        nhead=2,
        num_layers=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        embed_activation=nn.GELU(),
        time_inp_dim=5, # depends on the freq map | 't' minutes level = 5
    ):
        super(Predictor, self).__init__() 

        self.activation = embed_activation if embed_activation else nn.GELU()
        self.predictor_embed_dim = predictor_embed_dim

        self.predictor_embed = nn.Linear(
            encoder_embed_dim, predictor_embed_dim, bias=True
        )

        # Added representation
        self.pos_embed = PositionalEmbedding(predictor_embed_dim)
        self.time_embed = TimeFeatureEmbedding(predictor_embed_dim, time_inp_dim)

        # Mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Transformer part of the decoder
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
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

        # To normalize and map back to the encoder dimension (before applying the loss function)
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, encoder_embed_dim, bias=True
        )

    def forward(self, encoded_vals, x_mark, masks=None, non_masks=None):
        assert (masks is not None) and (encoded_vals is not None), "No input found"
        # encoded_vals: (B, L_ctx, D_enc) context-only reps (unmasked)
        # x_mark: (B, L_full, time_inp_dim) timestamps for full sequence
        # masks / non_masks: boolean or 0/1 masks over L_full (shape: B x L_full)

        B, L_full, patch_size, _time_inp = x_mark.size()
        _, L_ctx, _ = encoded_vals.size()               

        # map the output of the encoder to the predictor's dimension
        x = self.predictor_embed(encoded_vals)                              # (B, L_ctx, Dp)

        # build full-seq pos+time encodings
        pos_full = self.pos_embed(L_full).repeat(B, 1, 1)                   # (B, L_full, Dp)

        if x_mark is not None and not torch.allclose(x_mark, torch.zeros_like(x_mark)):
            tim_full = self.time_embed(x_mark)                               # (B, L_full, Dp)
            pos_tim_full = pos_full + tim_full
        else:
            pos_tim_full = pos_full

        # select encodings for ctx and masked locations
        ctx_pos_tim = apply_mask(pos_tim_full, non_masks)            # (B, L_ctx, Dp)
        x = x + ctx_pos_tim                                                 # add pos+time context tokens

        tgt_pos_tim = apply_mask(pos_tim_full, masks)                # (B, L_mask, Dp)
        pred_tokens = self.mask_token.repeat(B, tgt_pos_tim.size(1), 1)
        pred_tokens = pred_tokens + tgt_pos_tim                             # mask queries

        x = torch.cat([x, pred_tokens], dim=1)                              # (B, L_ctx+L_mask, Dp)

        for blk in self.predictor_blocks:
            x = blk(x, mask=None)
        
        x = self.predictor_norm(x)

        # Output only the part related to the masked area and adapt the dim
        # because the masked tokens were concatenated after the un-masked
        x = x[:, L_ctx:]                                                    # (B, L_mask, Dp)
        x = self.predictor_proj(x)                                          # (B, L_mask, D_enc)                                        
        
        return x