import pandas as pd
import torch
import random
import numpy as np
import os

from utils.timefeatures import time_features
from .base_loader.base_loader import CSVDataLoader, JSONDataLoader

class JEPALoader(CSVDataLoader):
    def __init__(self, *args, gluco_cache_path=None, **kwargs):
        """
        JEPA DataLoader with optional pre-computed glucodensity patches.
        
        Args:
            gluco_cache_path: Path to pre-computed glucodensity patches pickle file
            **kwargs: Other arguments passed to CSVDataLoader
        """
        super().__init__(*args, **kwargs)
        self.gluco_cache_path = gluco_cache_path
        self.gluco_cache = None
        
        # Load pre-computed cache if provided
        if gluco_cache_path and os.path.exists(gluco_cache_path):
            import pickle
            print(f"Loading pre-computed glucodensity patches from {gluco_cache_path}...")
            with open(gluco_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.gluco_cache = cache_data['gluco_patches']
                print(f"✅ Loaded {len(self.gluco_cache)} pre-computed samples")
        elif gluco_cache_path:
            print(f"⚠️  Warning: Cache path provided but file not found: {gluco_cache_path}")
    
    def __getitem__(self, idx):
        subject, split_idx = self.samples[idx]

        # get subject's data
        subject_df = self.df[self.df[self.subject_col] == subject]
        ts_raw = subject_df[self.glucose_value_col].values

        # normalize on-the-fly
        assert self.normalize and self.global_mean is not None and self.global_std is not None, "ERROR: No normalization done to load this sample"

        ts = (ts_raw - self.global_mean) / self.global_std
        ts = ts_raw
        ts = torch.tensor(ts).float()
        df_stamp = subject_df[[self.timestamp_col]]
        df_stamp["timestamp"] = pd.to_datetime(df_stamp.timestamp)
        timestamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        timestamp = torch.tensor(timestamp).float()

        # extract the split
        start_idx = split_idx * self.series_split_size
        end_idx = start_idx + self.series_split_size
        selected_series = ts[start_idx:end_idx]
        selected_timestamp = timestamp[start_idx:end_idx]

        if len(selected_series) < self.series_split_size:
            # add padding
            padding_size = self.series_split_size - len(selected_series)
            padding_glucose = np.full(padding_size, selected_series[-1])

            # create timestamp padding by adding 5 minutes interval
            last_timestamp = pd.Timestamp(selected_timestamp[-1])
            padding_timestamp = pd.date_range(
                start=last_timestamp + pd.Timedelta(minutes=5),
                periods=padding_size,
                freq='5min'
            )

            selected_series = np.concatenate([selected_series, padding_glucose])
            selected_timestamp = np.concatenate([selected_timestamp, padding_timestamp])

        assert selected_series.shape[0] == selected_timestamp.shape[0], "Size between glucose and timestamp don't match"
        assert selected_series.shape[0] == self.series_split_size, f"Expected {self.series_split_size} but got {selected_series.shape[0]}"
        assert selected_timestamp.shape[0] == self.series_split_size, f"Expected {self.series_split_size} but got {selected_timestamp.shape[0]}"

        # divide the selected smaller time series into patches
        num_patches = len(selected_series) // self.patch_size
        patches = [selected_series[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]
        timestamp_patches = [selected_timestamp[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]

        # convert patches to tensor
        patches_tensor = torch.stack(patches)
        timestamp_patches_tensor = torch.stack(timestamp_patches)

        # create the mask for the patches
        num_masked_patches = int(num_patches * self.mask_ratio)
        mask_indices = random.sample(range(num_patches), num_masked_patches) if num_masked_patches > 0 else []
        non_mask_indices = [i for i in range(num_patches) if i not in mask_indices]

        mask_indices = torch.tensor(mask_indices, dtype=torch.long) if len(mask_indices) > 0 else torch.empty(0, dtype=torch.long)
        non_mask_indices = torch.tensor(non_mask_indices, dtype=torch.long) if len(non_mask_indices) > 0 else torch.empty(0, dtype=torch.long)

        if not self.use_time_feature:
            time_feat = torch.zeros_like(timestamp_patches_tensor)
        else:
            time_feat = timestamp_patches_tensor

        # Load pre-computed glucodensity patches if available; else compute on-the-fly
        if self.gluco_cache is not None:
            subject, split_idx = self.samples[idx]
            try:
                gluco_patches = self.gluco_cache[(subject, split_idx)]
                gluco_patches = torch.tensor(gluco_patches, dtype=torch.float32)
                return patches_tensor, time_feat, mask_indices, non_mask_indices, gluco_patches
            except KeyError:
                # Cache was built for different samples (e.g. different CSV/split); compute from patches
                from utils.glucodensity_utils import compute_glucodensity_patches_from_cgm
                p = getattr(self, "gluco_spatial_patch_size", 8)
                g = getattr(self, "gluco_gridsize", 32)
                gluco_np = compute_glucodensity_patches_from_cgm(
                    patches_tensor.numpy(), patch_size=p, gridsize=g
                )
                gluco_patches = torch.tensor(gluco_np, dtype=torch.float32)
                return patches_tensor, time_feat, mask_indices, non_mask_indices, gluco_patches
        else:
            return patches_tensor, time_feat, mask_indices, non_mask_indices

class GluFormerDataLoader(CSVDataLoader):
    def glucose_to_bin(self, g):
        g = np.clip(g, self.min_glucose, self.max_glucose)
        width = (self.max_glucose - self.min_glucose) / self.num_bins

        bin_idx = np.floor((g - self.min_glucose) / width).astype(int) # min = 40, g = 60 -> 20
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1) # will use the num_bins-th for PAD
        return bin_idx

    def __getitem__(self, idx):
        subject, split_idx = self.samples[idx]

        # assume we already have the vocab_size
        subject_df = self.df[self.df[self.subject_col] == subject]
        ts = subject_df[self.glucose_value_col].values

        # no normalization, we take the value as token like a word
        start_idx = split_idx * self.series_split_size
        end_idx = start_idx + self.series_split_size
        selected_series = ts[start_idx:end_idx]

        # turn into bin token
        selected_series = [self.glucose_to_bin(int(s)) for s in selected_series]

        selected_series = np.array(selected_series)
        
        if len(selected_series) < self.series_split_size:
            # add padding
            padding_size = self.series_split_size - len(selected_series)
            padding_glucose = np.full(padding_size, self.num_bins) # last token is padding

            selected_series = np.concatenate([selected_series, padding_glucose])

        assert selected_series.shape[0] == self.series_split_size, f"Expected {self.series_split_size} but got {selected_series.shape[0]}"
        
        return selected_series
        
class ClassificationDataLoader(JSONDataLoader):
    def _extract_target(self, v):
        return v['y'][self.metabolic]['class']

    def _format_target(self, y):
        return torch.tensor(y, dtype=torch.long)
