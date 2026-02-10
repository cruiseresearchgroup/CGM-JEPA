import torch
import torch.nn as nn

from utils.embed import PositionalEmbedding
from utils.modules import *
from utils.mask_utils import apply_mask


class SpatialPatchEmbedding(nn.Module):
    """
    Vision Transformer-style patch embedding for glucose density images.
    Converts 2D spatial patches into tokens.
    """
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size  # Spatial patch size (patch_size x patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # For glucose density: each patch is (patch_size, patch_size, in_channels)
        # Flatten to (patch_size * patch_size * in_channels) and project
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, num_patches, patch_size, patch_size, in_channels) - 2D spatial patches
        Returns:
            x: (B, num_patches, embed_dim) - embedded patches
        """
        B, num_patches, patch_h, patch_w, in_channels = x.shape
        assert patch_h == self.patch_size and patch_w == self.patch_size, \
            f"Patch size mismatch: expected {self.patch_size}x{self.patch_size}, got {patch_h}x{patch_w}"
        
        # Flatten 2D spatial patch: (B, num_patches, patch_size * patch_size * in_channels)
        x = x.view(B, num_patches, patch_h * patch_w * in_channels)
        x = self.proj(x)
        return x


class GlucodensitySpatialEncoder(nn.Module):
    """
    Separate encoder dedicated to glucose density spatial representations.
    This encoder processes 2D KDE-based glucose density patches independently
    from the CGM temporal encoder.
    """
    def __init__(
        self,
        num_gluco_patches,
        gluco_patch_size,
        gluco_in_channels,
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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_gluco_patches = num_gluco_patches
        self.jepa = jepa

        # Spatial patch embedding
        self.gluco_embedding = SpatialPatchEmbedding(
            patch_size=gluco_patch_size,
            in_channels=gluco_in_channels,
            embed_dim=embed_dim
        )
        
        # Spatial positional embedding for glucose density
        self.gluco_pos_embed = PositionalEmbedding(embed_dim)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList(
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

        # DEPRECATED: Projection head (optional, for downstream tasks)
        self.proj = MLP(
            in_features=embed_dim,
            hidden_features=1024,
            out_features=48,
            act_layer=nn.GELU
        )

    def forward(self, gluco_patches, gluco_mask=None):
        """
        Args:
            gluco_patches: (B, num_gluco_patches, gluco_patch_size, gluco_patch_size, gluco_in_channels) 
                          - glucose density spatial patches
            gluco_mask: optional mask for glucose density patches (only used when self.jepa=True)
        
        Returns:
            encoded_tokens: (B, num_gluco_patches, embed_dim)
            proj_output: same shape as encoded_tokens
        """
        # Embed glucose density spatial patches
        gluco_emb = self.gluco_embedding(gluco_patches)  # (B, num_gluco_patches, embed_dim)
        
        # Add spatial positional embedding
        gluco_pos = self.gluco_pos_embed(gluco_emb.size(1))  # (1, num_gluco_patches, embed_dim)
        gluco_emb = gluco_emb + gluco_pos
        
        # Apply glucose density mask if provided (only when JEPA is enabled)
        if self.jepa and gluco_mask is not None:
            # gluco_mask should be (B, K) format for apply_mask
            gluco_emb = apply_mask(gluco_emb, [gluco_mask])
        
        # Process through transformer encoder
        x = gluco_emb
        for blk in self.encoder_blocks:
            x = blk(x, mask=None)
        
        x = self.encoder_norm(x)
        
        return x, self.proj(x)
