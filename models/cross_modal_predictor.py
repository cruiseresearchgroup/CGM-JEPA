import torch
import torch.nn as nn

from utils.embed import PositionalEmbedding
from utils.modules import *


class CrossModalPredictor(nn.Module):
    """
    Cross-modal predictor that takes CGM encoder context and predicts
    masked glucose density spatial patch embeddings.
    
    This enables the CGM encoder to learn representations that align with
    the distributional structure captured in glucose density KDEs.
    """
    def __init__(
        self,
        num_gluco_patches,
        encoder_embed_dim=96,  # CGM encoder embedding dimension
        predictor_embed_dim=48,
        nhead=2,
        num_layers=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        embed_activation=nn.GELU(),
    ):
        super(CrossModalPredictor, self).__init__() 

        self.activation = embed_activation if embed_activation else nn.GELU()
        self.predictor_embed_dim = predictor_embed_dim
        self.num_gluco_patches = num_gluco_patches

        # Project CGM encoder embeddings to predictor dimension
        self.predictor_embed = nn.Linear(
            encoder_embed_dim, predictor_embed_dim, bias=True
        )

        # Spatial positional embedding for glucose density patches
        self.pos_embed = PositionalEmbedding(predictor_embed_dim)

        # Mask tokens for glucose density patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Transformer blocks
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

        # Normalize and map back to encoder dimension
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, encoder_embed_dim, bias=True
        )

    def forward(self, cgm_context, gluco_masks=None, gluco_non_masks=None):
        """
        Predict masked glucose density embeddings from CGM encoder context.
        
        Args:
            cgm_context: (B, L_cgm, D_enc) - CGM encoder context embeddings (unmasked CGM patches)
            gluco_masks: (B, L_gluco) - boolean mask for masked glucose density patches
            gluco_non_masks: (B, L_gluco) - boolean mask for unmasked glucose density patches
        
        Returns:
            predicted_gluco_emb: (B, L_gluco_mask, D_enc) - predicted glucose density embeddings
        """
        assert (gluco_masks is not None) and (cgm_context is not None), "No input found"
        
        B, L_cgm, D_enc = cgm_context.size()
        L_gluco = self.num_gluco_patches

        # Map CGM encoder output to predictor dimension
        x = self.predictor_embed(cgm_context)  # (B, L_cgm, Dp)

        # Build spatial positional encodings for glucose density patches
        pos_gluco = self.pos_embed(L_gluco).repeat(B, 1, 1)  # (B, L_gluco, Dp)

        # Select encodings for context (unmasked) and masked locations
        # For cross-modal prediction, we use all CGM context tokens
        # and predict all glucose density patches (both masked and unmasked)
        # But we only compute loss on masked patches
        
        # Create mask tokens for all glucose density patches
        pred_tokens = self.mask_token.repeat(B, L_gluco, 1)  # (B, L_gluco, Dp)
        pred_tokens = pred_tokens + pos_gluco  # Add positional encoding

        # Concatenate CGM context with glucose density mask tokens
        x = torch.cat([x, pred_tokens], dim=1)  # (B, L_cgm + L_gluco, Dp)

        # Process through transformer
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)
        
        x = self.predictor_norm(x)

        # Extract only the glucose density predictions (last L_gluco tokens)
        x = x[:, L_cgm:]  # (B, L_gluco, Dp)
        
        # Apply mask to get only masked predictions for loss computation
        # gluco_masks is (B, K) where K is the number of masked patches
        if gluco_masks is not None:
            # Use gather to select masked patches: (B, L_gluco, Dp) -> (B, K, Dp)
            B, L_gluco, Dp = x.shape
            K = gluco_masks.size(1)  # Number of masked patches
            mask_keep = gluco_masks.unsqueeze(-1).repeat(1, 1, Dp)  # (B, K, Dp)
            x = torch.gather(x, dim=1, index=mask_keep)  # (B, K, Dp)
        
        # Project back to encoder dimension
        x = self.predictor_proj(x)  # (B, L_gluco_mask, D_enc)
        
        return x
