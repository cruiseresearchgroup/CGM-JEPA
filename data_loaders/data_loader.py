from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from .data_class import GluFormerDataLoader, JEPALoader, ClassificationDataLoader

import numpy as np

def get_jepa_loaders(path, batch_size, patch_size, use_time_feature, mask_ratio=0.9, gluco_cache_path=None):
    """
    Get JEPA data loaders with optional pre-computed glucodensity patches.
    
    Args:
        path: Path to CSV data file
        batch_size: Batch size
        patch_size: Patch size
        use_time_feature: Whether to use time features
        mask_ratio: Mask ratio for JEPA
        gluco_cache_path: Path to pre-computed glucodensity patches (optional)
    """
    jepa_loader = JEPALoader(
        path_data=path,
        series_split_size=288,
        patch_size=patch_size, # 1 hour
        mask_ratio=mask_ratio,
        use_time_feature=use_time_feature,
        gluco_cache_path=gluco_cache_path
    )

    dataloader = DataLoader(
        jepa_loader,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader

def get_gluformer_dataloader(path, batch_size):
    gluformer_dataloader = GluFormerDataLoader(
        path_data=path,
        series_split_size=288,
    )

    dataloader = DataLoader(
        gluformer_dataloader,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader

def get_eval_loaders(path, patch_size, task, extract_method, metabolic, seed,
    batch_size=32, global_mean=None, global_std=None, y_global_mean=None,
    y_global_std=None, portion=1.0, output_type="patch", context_size=1
):
    '''
        @brief: get the data used for evaluation
        @return: data loader with (x, y) where y is the subphenotype label
    '''

    loader = ClassificationDataLoader(
        path_data=path,
        patch_size=patch_size,
        metabolic=metabolic,
        extract_method=extract_method,
        global_mean=global_mean,
        global_std=global_std,
        output_type=output_type
    )
    
    dataloader = DataLoader(
        loader,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader

def stratify_split_loader(
    original_loader: DataLoader,
    n_splits: int,
    batch_size: int,
    seed: int,
    train_portion: float = 1.0,  # portion to use for the training (after split)
    test_portion: float = 1.0,   # portion to use for the test (after split) 
):
    def _stratified_subsample(split_indices, portion, rng_seed):
        """Take a stratified subset of the provided indices."""
        if portion >= 1.0:
            return np.array(split_indices)

        subset_labels = labels[split_indices]
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=portion, random_state=rng_seed
        )
        selected_idx, _ = next(sss.split(np.zeros(len(split_indices)), subset_labels))
        return np.array(split_indices)[selected_idx]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    dataset = original_loader.dataset
    indices = np.arange(len(dataset))

    labels = np.array([dataset.samples[i][3] for i in indices])

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), 1):
        # pick stratified train/test indices: n-shot (per class) or portion

        train_idx = _stratified_subsample(train_idx, train_portion, seed + fold)
        test_idx = _stratified_subsample(test_idx, test_portion, seed + fold)

        # make the new subset from original loader
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

        yield  fold, train_loader, test_loader