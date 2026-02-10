import torch
import os
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore")

from root import PROJECT_ROOT

from config.config_downstream import config
from config.model_configs import get_model_configs

from data_loaders.data_loader import get_eval_loaders, stratify_split_loader

from utils.main_utils import analyze_results, convert_numpy_types

from eval.training.trainer import Trainer

from config.config_downstream import config


def main():
    pipeline = config["pipeline"]
    pipeline_logs = {}

    # current available pipelines, can be adapted in the future with more datasets as well
    if pipeline == "initial_to_validation":

        initial_cohort_loader = get_eval_loaders(
            path=config["path_data"],
            patch_size=config["patch_size"],
            task=config["task"],
            extract_method=config["extract_method"],
            metabolic=config["metabolic"],
            seed=config["random_seed"],
            batch_size=config["batch_size"]
        ) 

        for i in range(config["num_iterations"]):
            # Initialize nested dict for this iteration
            if i not in pipeline_logs:
                pipeline_logs[i] = {}
            
            # Fixed iteration seed
            iteration_seed = config["random_seed"] * 1000 + i
            
            for fold, train_loader, test_loader in stratify_split_loader(
                original_loader=initial_cohort_loader,
                n_splits=config["n_splits"],
                batch_size=config["batch_size"],
                seed=iteration_seed,
                train_portion=config["train_portion"],
                test_portion=config["test_portion"]
            ):
                # Compute stats on training split only to prevent leakage
                train_indices = train_loader.dataset.indices
                train_loader.dataset.dataset.compute_stats(train_indices, normalize_x=True, normalize_y=False)

                # NOTE: test_loader shares the same reference with train_loader
                # so the global_mean and global_std have already been updated

                # Use unique seed combining iteration and fold for validation sampling
                validation_seed = iteration_seed + fold * 100

                # Load the validation cohort loader with the current train distribution                
                validation_cohort_loader = get_eval_loaders(
                    path=config["val_path_data"],
                    patch_size=config["patch_size"],
                    task=config["task"],
                    extract_method=config["val_extract_method"], # for validation, we have 4 different ways of extraction
                    metabolic=config["metabolic"],
                    seed=validation_seed,  # Unique seed per iteration and fold
                    batch_size=config["batch_size"],
                    global_mean=train_loader.dataset.dataset.global_mean,
                    global_std=train_loader.dataset.dataset.global_std,
                    y_global_mean=None,
                    y_global_std=None,
                    portion=config["val_portion"]
                )

                model_configs = get_model_configs(config)
                trainer = Trainer(
                    task=config["task"],
                    model_configs=model_configs,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    val_loader=validation_cohort_loader
                )
                results = trainer.execute()
                # Store results in nested structure: iteration -> fold
                pipeline_logs[i][fold] = results
    elif pipeline == "validation_to_validation":
        # Create two loaders from the same dataset with different extract_methods
        # Training loader uses extract_method, test loader uses val_extract_method
        train_cohort_loader = get_eval_loaders(
            path=config["val_path_data"],
            patch_size=config["patch_size"],
            task=config["task"],
            extract_method=config["extract_method"],  # Use extract_method for training
            metabolic=config["metabolic"],
            seed=config["random_seed"],
            batch_size=config["batch_size"]
        )

        for i in range(config["num_iterations"]):
            # Initialize nested dict for this iteration
            if i not in pipeline_logs:
                pipeline_logs[i] = {}
            
            # Use unique seed for each iteration: base_seed * 1000 + iteration_number
            # This ensures large gaps between iterations for truly independent sampling
            iteration_seed = config["random_seed"] * 1000 + i
            
            # Split training loader to get train/test indices
            for fold, train_loader_temp, test_loader_temp in stratify_split_loader(
                original_loader=train_cohort_loader,
                n_splits=config["n_splits"],
                batch_size=config["batch_size"],
                seed=iteration_seed,  # Unique seed per iteration
                train_portion=config["train_portion"],
                test_portion=config["test_portion"]
            ):
                # Extract indices from the split
                train_indices = train_loader_temp.dataset.indices
                test_indices = test_loader_temp.dataset.indices
                
                # Compute stats on training split only (using train_cohort_loader with extract_method)
                train_cohort_loader.dataset.compute_stats(train_indices, normalize_x=True, normalize_y=False)
                
                # Create train loader from train_cohort_loader (uses extract_method)
                from torch.utils.data import Subset, DataLoader
                train_subset = Subset(train_cohort_loader.dataset, train_indices)
                train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
                
                # Create test loader from validation dataset with val_extract_method
                # Use normalization stats from training split
                # Use unique seed combining iteration and fold for test sampling
                test_seed = iteration_seed + fold * 100
                test_cohort_loader = get_eval_loaders(
                    path=config["val_path_data"],
                    patch_size=config["patch_size"],
                    task=config["task"],
                    extract_method=config["val_extract_method"],  # Use val_extract_method for testing
                    metabolic=config["metabolic"],
                    seed=test_seed,  # Unique seed per iteration and fold
                    batch_size=config["batch_size"],
                    global_mean=train_cohort_loader.dataset.global_mean,
                    global_std=train_cohort_loader.dataset.global_std,
                    y_global_mean=None,
                    y_global_std=None
                )
                
                # Create test loader using test_indices from val_extract_method loader
                test_subset = Subset(test_cohort_loader.dataset, test_indices)
                test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=True)
                
                model_configs = get_model_configs(config)
                trainer = Trainer(
                    task=config["task"],
                    model_configs=model_configs,
                    train_loader=train_loader,
                    test_loader=test_loader
                )
                results = trainer.execute()
                # Store results in nested structure: iteration -> fold
                pipeline_logs[i][fold] = results


    # save the logs
    results_log = {
        "pipeline": config["pipeline"],
        "task": config["task"],
        "metabolic": config["metabolic"],
        "extract_method": config["extract_method"],
        "val_extract_method": config["val_extract_method"],
        "seed": config["random_seed"],
        "num_iterations": config["num_iterations"],
        "n_splits": config["n_splits"],
        "train_portion": config["train_portion"],
        "test_portion": config["test_portion"],
        "train_n_shot": config.get("train_n_shot"),
        "test_n_shot": config.get("test_n_shot"),
        "pipeline_logs": pipeline_logs
    }

    results_log = convert_numpy_types(results_log)
    
    # save the results
    path_save = config["path_save"]
    path_save = os.path.join(path_save, config["pipeline"])
    os.makedirs(path_save, exist_ok=True)
    path_name = os.path.join(path_save, "results.json")
    with open(path_name, 'w') as f:
        json.dump(results_log, f, indent=4)

    # analyze results and optionally log to wandb
    analyze_results(
        path_name,
        use_wandb=config.get("enable_wandb", False),
        wandb_project=config.get("wandb_project"),
        experiment_config=config  # Full config for reproducibility
    )

    return results_log

if __name__ == "__main__":
    main()
