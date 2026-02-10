import torch
import numpy as np
from utils.glucodensity_utils import compute_glucodensity_patches_from_cgm

class DataTransformer:
    def __init__(self, config):
        self.config = config

    def transform(self, x):
        raise NotImplementedError

    def encode(self, encoder, x):
        raise NotImplementedError

class PatchDataTransformer(DataTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.get("patch_size", 12)
        self.mean = config.get("mean", None)
        self.std = config.get("std", None)

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        # Normalize
        if self.mean is not None and self.std is not None:
            mask = x != -1
            x[mask] = (x[mask] - self.mean) / self.std
        
        # Patchify
        if len(x) % self.patch_size != 0:
            # add padding
            padding_size = self.patch_size - (len(x) % self.patch_size) 
            padding_x = np.full(padding_size, -1)
            x = np.concatenate([x, padding_x])
        
        num_patches = len(x) // self.patch_size
        x_patches = [x[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]
        x_patches_tensor = torch.tensor(np.array(x_patches)).float()
        return x_patches_tensor

    def encode(self, encoder, x, x_mark=None):
        emb, proj = encoder(x, x_mark)        
        return torch.mean(emb, dim=1)

class TokenDataTransformer(DataTransformer):
    def __init__(self, config):
        super().__init__(config)
        # From our csv CGM
        self.num_bins = 280
        self.min_glucose = 40
        self.max_glucose = 320 # based on max data and what can be measured by the device

    def glucose_to_bin(self, g):
        g = np.clip(g, self.min_glucose, self.max_glucose)
        width = (self.max_glucose - self.min_glucose) / self.num_bins
        bin_idx = np.floor((g - self.min_glucose) / width).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1) # clip bin idx greater than what we know
        return bin_idx

    def transform(self, x):
        x_tokenized = [self.glucose_to_bin(int(s)) for s in x]
        x_tokenized = np.array(x_tokenized)
        return x_tokenized

    def encode(self, encoder, x):
        device = next(encoder.parameters()).device
        x = x.to(device).long()
        encoded = encoder(x)
        return torch.mean(encoded, dim=1)

class FlatDataTransformer(DataTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.mean = config.get("mean", None)
        self.std = config.get("std", None)

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        # Normalize | for TSFM this will be None and skip normalization
        if self.mean is not None and self.std is not None:
            mask = x != -1
            x[mask] = (x[mask] - self.mean) / self.std
        return torch.tensor(x).float()

    def encode(self, encoder, x):
        # Flat input, just pass through.
        # Some encoders (e.g., JEPA-style or TS2Vec wrappers) return (emb, proj),
        # where we only want the embedding tensor.
        output = encoder(x)
        if isinstance(output, tuple):
            output = output[0]
        return output
