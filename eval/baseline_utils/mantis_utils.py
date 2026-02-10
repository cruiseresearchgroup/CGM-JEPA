"""
Shared Mantis (time series foundation model) utilities for model_configs and run_mantis_class_reg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MANTIS_SEQ_LENGTH = 512


class MantisEncoderWrapper(nn.Module):
    """
    Wraps MantisTrainer so it can be used as an encoder in the class_reg Trainer.
    Interpolates flat time series to (num_samples, num_channels, seq_len), then calls model.transform(x).
    """

    def __init__(self, mantis_trainer, device: str):
        super().__init__()
        self.mantis_trainer = mantis_trainer
        self.device = device
        self.dummy_param = nn.Parameter(torch.zeros(1).to(device))

    def forward(self, x: torch.Tensor) -> tuple:
        # Target shape: (num_samples, num_channels, seq_len) = (B, C, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        num_samples, num_channels, seq_len = x.shape
        if seq_len != MANTIS_SEQ_LENGTH:
            x = F.interpolate(
                x.float(),
                size=MANTIS_SEQ_LENGTH,
                mode="linear",
                align_corners=False,
            )
        with torch.no_grad():
            emb = self.mantis_trainer.transform(x)
        if isinstance(emb, torch.Tensor):
            pass
        else:
            emb = torch.tensor(emb, dtype=torch.float32, device=self.device)
        if emb.device != self.device:
            emb = emb.to(self.device)
        return emb, emb


def load_mantis_encoder(device) -> MantisEncoderWrapper:
    """Load Mantis-8M and return wrapper for use in model_configs."""
    from mantis.architecture import Mantis8M
    from mantis.trainer import MantisTrainer

    mantis_device = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    network = Mantis8M(device=mantis_device)
    network = network.from_pretrained("paris-noah/Mantis-8M")
    mantis_trainer = MantisTrainer(device=mantis_device, network=network)
    wrapper = MantisEncoderWrapper(mantis_trainer, mantis_device)
    return wrapper.to(device)
