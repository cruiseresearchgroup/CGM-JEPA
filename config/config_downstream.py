from root import PROJECT_ROOT

config = {
    # Data Path
    "path_data": f"{PROJECT_ROOT}/Dataset_Open/train_split.json", # Initial cohort data
    "val_path_data": f"{PROJECT_ROOT}/Dataset_Open/validation_split.json",

    # Save Path
    "path_save" : f"{PROJECT_ROOT}/logs/outputmodel/",

    "wandb_run_name": None,  # Optional: set to customize run name
    "wandb_project": "rebuttal_classification",  # wandb project name
    "wandb_tags": ["pretrained_with_open_dataset"],  # extra tags appended to base tags
    "enable_wandb": True, # set to True to enable logging

    "random_seed": 42,

    # Data loader
    "patch_size": 12, # 5 mins x 12 = 1 hour

    # Pipeline
    "pipeline": "initial_to_validation", # initial_to_validation, validation_to_validation
    "tag": "Re running with statistical significance", # some notes

    # Cross Validation
    "n_splits": 2,

    "task": "classification", # classification

    # Classification
    "extract_method": "ctru_venous", 
    "val_extract_method": "ctru_venous", # ctru_venous, ctru_cgm, home_cgm_1, home_cgm_2, cgm_home_mean, cgm_all_mean 
    "metabolic": "ir", # ir or beta
    "num_class": 1,
    "use_encoder": True,
    "flatten": True, # classical models
    "num_iterations": 20, # iterations to repeat cv with different random seeds (2x20 = 40 paired obs)
    "linear_probe_only": True, # only use L2 logistic regression as classifier

    "train_portion": 1.0,
    "test_portion": 1.0,
    "val_portion": 1.0,

    # Encoder Version
    "cgm_jepa_version": "v18",
    "cgm_jepa_glu_version": "v19", 
    "gluformer_version": "v5",
    "cgm_ts2vec_version": "v2",
    "cgm_mantis_version": "default",  # Set to "default" to enable, None to disable
    "cgm_moment_version": "default",  # Set to "default" to enable, None to disable

    # Optim for epoch based training with decoder
    "num_epochs": 100, 
    "batch_size" : 32,
    "lr": 1e-06,
}
