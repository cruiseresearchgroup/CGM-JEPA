import torch
import os
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

from root import PROJECT_ROOT

from config.config_downstream import config
from config.model_configs import get_model_configs

from data_loaders.data_loader import get_eval_loaders, stratify_split_loader

from utils.main_utils import analyze_results, convert_numpy_types

from eval.training.trainer import Trainer

from config.config_downstream import config


def compute_separation_metrics(model_configs, loader, device):
    """
    Compute separation metrics (Silhouette, CH, DB) on the FULL dataset
    for each encoder-based model config. No CV needed — just encode all
    samples and measure cluster quality.
    """
    results = {}

    for cfg in model_configs:
        name = cfg.get("name", "")
        encoder = cfg.get("encoder", None)
        output_type = cfg.get("output_type", "patch")
        normalize_for_encoder = cfg.get("normalize_for_encoder", True)

        if encoder is None:
            # PCA baseline — use raw flattened features
            all_x, all_y = [], []
            for subject, x, ctx_ts, y in loader:
                all_x.append(x.numpy().reshape(x.shape[0], -1))
                all_y.append(y.numpy())
            X = np.concatenate(all_x)
            y = np.concatenate(all_y)

            from sklearn.decomposition import PCA
            X = PCA(n_components=2).fit_transform(X)
        else:
            # Encode all samples with frozen encoder
            encoder.eval()
            all_emb, all_y = [], []
            with torch.no_grad():
                for subject, x, ctx_ts, y in loader:
                    x = x.to(device)
                    B = x.shape[0]

                    if output_type == "token":
                        # GluFormer: denormalize then tokenize to int bins.
                        # vocab_size comes from the encoder artifact — older GluFormer
                        # checkpoints used 278, current default is 280.
                        vocab_size = getattr(encoder, "vocab_size", 280)
                        x_flat = x.view(B, -1)
                        x_raw = x_flat * loader.dataset.global_std + loader.dataset.global_mean
                        x_clipped = torch.clamp(x_raw, 40, 320)
                        width = (320 - 40) / vocab_size
                        x_tokens = torch.floor((x_clipped - 40) / width).long().clamp(0, vocab_size - 1).to(device)
                        emb = encoder(x_tokens)
                    elif output_type == "flat":
                        x_flat = x.view(B, -1)
                        if not normalize_for_encoder:
                            x_flat = x_flat * loader.dataset.global_std + loader.dataset.global_mean
                        out = encoder(x_flat.to(device))
                        emb = out[0] if isinstance(out, tuple) else out
                    else:
                        # patch (CGM-JEPA / X-CGM-JEPA)
                        out = encoder(x)
                        emb = out[0] if isinstance(out, tuple) else out

                    if emb.dim() == 3:
                        emb = torch.mean(emb, dim=1)
                    all_emb.append(emb.cpu().numpy())
                    all_y.append(y.numpy())
            X = np.concatenate(all_emb)
            y = np.concatenate(all_y)

        if len(np.unique(y)) < 2:
            continue

        try:
            unique_labels = np.unique(y)
            n_classes = len(unique_labels)
            kmeans_labels = KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit_predict(X)
            metrics = {
                "silhouette": float(silhouette_score(X, y)),
                "calinski_harabasz": float(calinski_harabasz_score(X, y)),
                "davies_bouldin": float(davies_bouldin_score(X, y)),
                "ari": float(adjusted_rand_score(y, kmeans_labels)),
                "nmi": float(normalized_mutual_info_score(y, kmeans_labels)),
            }

            # Between/within class distance ratio (scatter form)
            centroids = []
            within_ss = 0.0
            for c in unique_labels:
                X_c = X[y == c]
                if X_c.shape[0] == 0:
                    continue
                mu_c = X_c.mean(axis=0)
                centroids.append(mu_c)
                diffs = X_c - mu_c
                within_ss += float(np.sum(diffs * diffs))

            if len(centroids) > 1:
                centroids = np.stack(centroids, axis=0)
                global_mu = X.mean(axis=0)
                diffs_c = centroids - global_mu
                between_ss = float(np.sum(diffs_c * diffs_c) * (X.shape[0] / n_classes))
                ratio = between_ss / within_ss if within_ss > 0 and np.isfinite(between_ss) else float("nan")
                metrics["between_within_ratio"] = float(ratio)
            else:
                metrics["between_within_ratio"] = float("nan")

            # Intra-cluster (avg distance to centroid) & inter-cluster (pairwise centroid distance)
            try:
                intra_distances = []
                for c in unique_labels:
                    X_c = X[y == c]
                    if X_c.shape[0] < 2:
                        continue
                    mu_c = X_c.mean(axis=0)
                    dists = np.linalg.norm(X_c - mu_c, axis=1)
                    intra_distances.append(np.mean(dists))

                inter_distances = []
                if len(centroids) > 1:
                    for i in range(len(centroids)):
                        for j in range(i + 1, len(centroids)):
                            inter_distances.append(np.linalg.norm(centroids[i] - centroids[j]))

                metrics["intra_cluster_distance"] = float(np.mean(intra_distances)) if intra_distances else float("nan")
                metrics["inter_cluster_distance"] = float(np.mean(inter_distances)) if inter_distances else float("nan")
            except Exception:
                metrics["intra_cluster_distance"] = float("nan")
                metrics["inter_cluster_distance"] = float("nan")

            metrics["n_samples"] = len(y)
            results[name] = metrics
        except Exception as e:
            print(f"  Warning: separation metrics failed for {name}: {e}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_overrides", type=str, default=None,
                        help="Path to JSON file with config overrides")
    args, _ = parser.parse_known_args()

    if args.ablation_overrides:
        with open(args.ablation_overrides) as f:
            overrides = json.load(f)
        config.update(overrides)
        print(f"Applied overrides: {list(overrides.keys())}")

    pipeline = config["pipeline"]
    pipeline_logs = {}
    all_preds_rows = []  # per-subject test predictions across all (iter, fold, model)

    # Load model configs once (downloads artifacts)
    model_configs = get_model_configs(config)

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

                trainer = Trainer(
                    task=config["task"],
                    model_configs=model_configs,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    val_loader=validation_cohort_loader
                )
                results = trainer.execute()
                for r in trainer.preds_rows:
                    all_preds_rows.append({"iteration": i, "fold": fold, **r})
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
                    y_global_std=None,
                )
                
                # Create test loader using test_indices from val_extract_method loader
                test_subset = Subset(test_cohort_loader.dataset, test_indices)
                test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=True)
                
                trainer = Trainer(
                    task=config["task"],
                    model_configs=model_configs,
                    train_loader=train_loader,
                    test_loader=test_loader
                )
                results = trainer.execute()
                for r in trainer.preds_rows:
                    all_preds_rows.append({"iteration": i, "fold": fold, **r})
                # Store results in nested structure: iteration -> fold
                pipeline_logs[i][fold] = results


    # Compute separation metrics on the FULL dataset (no CV)
    print("\nComputing separation metrics on full dataset...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    if pipeline == "initial_to_validation":
        # Use initial cohort with global normalization
        sep_loader = get_eval_loaders(
            path=config["path_data"], patch_size=config["patch_size"],
            task=config["task"], extract_method=config["extract_method"],
            metabolic=config["metabolic"], seed=config["random_seed"],
            batch_size=config["batch_size"]
        )
        sep_loader.dataset.compute_stats(normalize_x=True)
        sep_loader = torch.utils.data.DataLoader(sep_loader.dataset, batch_size=config["batch_size"], shuffle=False)
    else:
        # Use validation cohort
        sep_loader = get_eval_loaders(
            path=config["val_path_data"], patch_size=config["patch_size"],
            task=config["task"], extract_method=config["extract_method"],
            metabolic=config["metabolic"], seed=config["random_seed"],
            batch_size=config["batch_size"]
        )
        sep_loader.dataset.compute_stats(normalize_x=True)
        sep_loader = torch.utils.data.DataLoader(sep_loader.dataset, batch_size=config["batch_size"], shuffle=False)

    separation_results = compute_separation_metrics(model_configs, sep_loader, device)

    print("\nSeparation Metrics (full dataset):")
    print("=" * 60)
    for name, metrics in sorted(separation_results.items(), key=lambda x: x[1].get("silhouette", 0), reverse=True):
        print(f"  {name}:")
        print(f"    Silhouette: {metrics['silhouette']:.4f}  CH: {metrics['calinski_harabasz']:.4f}  DB: {metrics['davies_bouldin']:.4f}  ARI: {metrics.get('ari', float('nan')):.4f}  NMI: {metrics.get('nmi', float('nan')):.4f}")

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
        "pipeline_logs": pipeline_logs,
        "separation_metrics": separation_results,
    }

    results_log = convert_numpy_types(results_log)
    
    # save the results
    path_save = config["path_save"]
    path_save = os.path.join(path_save, config["pipeline"])
    os.makedirs(path_save, exist_ok=True)
    path_name = os.path.join(path_save, "results.json")
    with open(path_name, 'w') as f:
        json.dump(results_log, f, indent=4)

    # Per-subject test predictions, tagged with run context — for demographic
    # stratification and other post-hoc analyses.
    if all_preds_rows:
        import pandas as pd
        run_tag = config.get("wandb_run_name") or f"{config['pipeline']}_{config['extract_method']}_to_{config['val_extract_method']}_{config['metabolic']}"
        preds_dir = os.path.join("logs", "preds")
        os.makedirs(preds_dir, exist_ok=True)
        preds_path = os.path.join(preds_dir, f"{run_tag}.csv")
        df = pd.DataFrame(all_preds_rows)
        df["pipeline"] = config["pipeline"]
        df["metabolic"] = config["metabolic"]
        df["extract_method"] = config["extract_method"]
        df["val_extract_method"] = config["val_extract_method"]
        df["train_portion"] = config["train_portion"]
        df.to_csv(preds_path, index=False)
        print(f"Saved per-subject predictions: {preds_path}  ({len(df)} rows)")

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
