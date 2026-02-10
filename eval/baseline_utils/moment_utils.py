"""
Shared MOMENT (time series foundation model) utilities for run_moment_class_reg.

MOMENT expects input shape (batch, n_channels, context_length) = (B, 1, 512).
See: https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/classification.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MOMENT_SEQ_LENGTH = 512


class MOMENTEncoderWrapper(nn.Module):
    """
    Wraps MOMENTPipeline so it can be used as an encoder in the class_reg Trainer.
    Brings flat time series to (B, 1, 512) per MOMENT's expected input:
    - If seq_len < 512: left-pad to 512 and pass input_mask (zeros at padded positions).
    Then calls model(x_enc=x[, input_mask]) and returns output.embeddings.
    """

    def __init__(self, moment_pipeline, device: str):
        super().__init__()
        self.moment_pipeline = moment_pipeline
        self.device = device
        self.dummy_param = nn.Parameter(torch.zeros(1).to(device))

    def forward(self, x: torch.Tensor) -> tuple:
        dev = self.dummy_param.device
        x = x.to(dev)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        num_samples, num_channels, seq_len = x.shape
        input_mask = None

        # Shorter series: left-pad to 512 and use input_mask (zeros = padded, ones = attended).
        pad_left = MOMENT_SEQ_LENGTH - seq_len
        x = F.pad(x.float(), (pad_left, 0), mode="constant", value=0.0)
        input_mask = torch.ones(
            num_samples, MOMENT_SEQ_LENGTH, dtype=torch.float32, device=dev
        )
        input_mask[:, :pad_left] = 0

        with torch.no_grad():
            output = self.moment_pipeline(x_enc=x, input_mask=input_mask)

        emb = output.embeddings
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32, device=dev)
        emb = emb.to(dev)
        # Trainer / FlatDataTransformer expect (emb, emb) or single tensor
        return emb, emb


def load_moment_encoder(device, model_name: str = "AutonLab/MOMENT-1-large") -> MOMENTEncoderWrapper:
    """Load MOMENT in embedding mode and return wrapper for use in class_reg."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "embedding",
            "n_channels": 1,
        },
    )
    model.init()
    model.eval()
    wrapper = MOMENTEncoderWrapper(model, str(device))
    return wrapper.to(device)
