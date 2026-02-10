import numpy as np
import pandas as pd
import torch
import json
import random
from ..data_transformer import PatchDataTransformer, TokenDataTransformer, FlatDataTransformer

class JSONDataLoader():
    '''
        @brief: Base class for loading .json dataset, turning x into patches, and providing y
    '''
    def __init__(
        self,
        path_data,
        patch_size=12,
        metabolic='ir', # classification task | ir -> predict insulin resistance | beta -> beta-cell dysfunction
        extract_method='ctru_venous', # options: ctru_venous, home_cgm_2, home_cgm_1, ctru_cgm, cgm_home_mean, cgm_all_mean 
        use_time_feature=False,
        timeenc=1,
        freq='t',
        y_global_mean=None,
        y_global_std=None,
        global_mean=None, # supply this when evaluating to use distribution from training set
        global_std=None,  # supply this when evaluating to use distribution from training set
        min_glucose=None,
        max_glucose=None,
        num_bins=None,
        output_type="patch"
    ):
        with open(path_data, 'r') as f:
            dataset = json.load(f)
        
        # scaling for regression target
        self.y_global_mean = y_global_mean
        self.y_global_std = y_global_std

        self.global_mean = global_mean
        self.global_std = global_std
        self.patch_size = patch_size
        self.metabolic = metabolic

        self.normalize_x = (self.global_mean is not None and self.global_std is not None)
        self.normalize_y = (self.y_global_mean is not None and self.y_global_std is not None)
        
        self.min_glucose = min_glucose
        self.max_glucose = max_glucose
        self.num_bins = num_bins
        self.output_type = output_type

        self._init_transformer()

        samples = []
        # aggregate the data between subject
        for k, v in dataset.items():
            x = np.array(v['x'].get(extract_method, None), dtype=float)
            if not x.size:
                continue
            
            # store raw data
            y = self._extract_target(v)
            strat_label = v['y'][metabolic]['class']
            samples.append((k, x.tolist(), y, strat_label))

        self.samples = samples

    def _init_transformer(self):
        # For flat output_type, skip normalization when normalize_for_encoder is False
        # (e.g. Mantis, MOMENT expect raw time series; same idea as token for GluFormer)
        normalize_for_encoder = getattr(self, 'normalize_for_encoder', True)
        if self.output_type == "flat" and not normalize_for_encoder:
            flat_mean, flat_std = None, None
        else:
            flat_mean = self.global_mean if self.normalize_x else None
            flat_std = self.global_std if self.normalize_x else None

        config = {
            "patch_size": self.patch_size,
            "mean": self.global_mean if self.normalize_x else None,
            "std": self.global_std if self.normalize_x else None,
            "num_bins": self.num_bins,
            "min_glucose": self.min_glucose,
            "max_glucose": self.max_glucose,
        }
        
        if self.output_type == "patch":
            self.transformer = PatchDataTransformer(config)
        elif self.output_type == "token":
            self.transformer = TokenDataTransformer(config)
        else:
            # FlatDataTransformer: pass no stats when encoder expects raw (Mantis, MOMENT)
            flat_config = {**config, "mean": flat_mean, "std": flat_std}
            self.transformer = FlatDataTransformer(flat_config)
    
    def update_transformer(self, config):
        self.output_type = config.get("output_type", None)
        self.normalize_for_encoder = config.get("normalize_for_encoder", True)
        self._init_transformer()

    def _extract_target(self, v):
        raise NotImplementedError

    def _format_target(self, y):
        raise NotImplementedError

    def compute_stats(self, indices=None, normalize_x=False, normalize_y=False):
        '''
            @brief: Compute global mean and std from samples at given indices.
                    If indices is None, use all samples.
                    Only computes x stats if normalize_x is True.
                    Only computes y stats if normalize_y is True.
        '''
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

        if indices is None:
            indices = range(len(self.samples))
            
        all_vals = []
        all_targets = []
        for idx in indices:
            _, x_list, y, _ = self.samples[idx]
            if self.normalize_y:
                target_arr = np.array(y, dtype=float)
                all_targets.append(target_arr)
            if self.normalize_x:
                arr = np.array(x_list, dtype=float)
                valid = arr[arr != -1] # skip missing value in alignment
                if valid.size:
                    all_vals.append(valid)
        
        # Compute input statistics only if normalize_x is True
        if self.normalize_x:
            stacked = np.concatenate(all_vals)
            self.global_mean = stacked.mean()
            self.global_std = stacked.std()

            # set stats only for data transformer that needs to normalized the x
            if hasattr(self.transformer, "set_stats"):
                self.transformer.set_stats(self.global_mean, self.global_std)    

        # Compute target statistics only if normalize_y is True
        if self.normalize_y:
            all_targets = np.array(all_targets)
            self.y_global_mean = all_targets.mean()
            self.y_global_std = all_targets.std()

    def __getitem__(self, idx):
        subject, x_list, y, _ = self.samples[idx]
        x = np.array(x_list, dtype=float)
        
        # Transform x using the transformer
        # The transformer handles normalization and patching/tokenization
        x_transformed = self.transformer.transform(x)
        
        return subject, x_transformed, np.zeros_like(x_transformed), self._format_target(y)


    def __len__(self):
        return len(self.samples)

class CSVDataLoader():
    def __init__(
        self,
        path_data,
        batch_size=32,
        series_split_size=288, # daily data
        patch_size=12, 
        mask_ratio=0.25,
        use_time_feature=True, # whether we want to return the timestamp feature
        timeenc=1, # frequency based
        freq='t', # minutes level
    ):
        timestamp_col = "timestamp"
        glucose_value_col = "glucose_value"
        subject_col = "subject"
        df = pd.read_csv(
            path_data,
            parse_dates=[timestamp_col],
            low_memory=False,
            sep=","
        )
        
        # find minimum and maximum glucose values (excluding 'Low' and 'High')
        glucose_numeric = pd.to_numeric(df[glucose_value_col], errors='coerce')
        self.min_glucose = glucose_numeric.min()
        self.max_glucose = glucose_numeric.max()
        df.loc[df[glucose_value_col] == 'Low', glucose_value_col] = self.min_glucose
        df.loc[df[glucose_value_col] == 'High', glucose_value_col] = self.max_glucose
        df[glucose_value_col] = pd.to_numeric(df[glucose_value_col])
        df[glucose_value_col].apply(pd.to_numeric, downcast="float")

        # save num_bins
        self.num_bins = int(self.max_glucose - self.min_glucose)

        df.sort_values(by=[subject_col, timestamp_col], inplace=True)

        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.use_time_feature = use_time_feature

        # pre-compute all valid samples (subject, split_idx) pairs
        self.raw_samples = {}  # for storing all series for each subject without any series split size
        self.samples = []
        for subject in df[subject_col].unique():
            subject_df = df[df[subject_col] == subject]
            ts = subject_df[glucose_value_col].values
            self.raw_samples[subject] = ts
            num_splits = len(ts) // series_split_size
            for split_idx in range(num_splits):
                self.samples.append((subject, split_idx))

        self.df = df
        self.timestamp_col = timestamp_col
        self.glucose_value_col = glucose_value_col
        self.subject_col = subject_col
        self.subjects = df[subject_col].unique().tolist()
        self.timeenc = timeenc
        self.freq = freq


    def compute_stats(self, indices=None, normalize_x=False, normalize_y=None):
        '''
            @brief: Compute global mean and std from samples at given indices.
                    If indices is None, use all samples.
        '''
        self.normalize = normalize_x

        if self.normalize:
            if indices is None:
                all_series = np.concatenate(list(self.raw_samples.values()))
                self.global_mean = all_series.mean()
                self.global_std = all_series.std()
                return 
            
            unique_subjects = set()
            for i in indices:
                subject, _ = self.samples[i]
                unique_subjects.add(subject)
            
            all_series = []
            added_subjects = set()
            for subject in unique_subjects:
                if subject not in added_subjects:
                    all_series.append(self.raw_samples[subject])
            
            all_series = np.concatenate(all_series)
            self.global_mean = all_series.mean()
            self.global_std = all_series.std()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        # number of smaller time series created from the full series
        return len(self.samples)