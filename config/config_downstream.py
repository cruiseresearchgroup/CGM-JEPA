from root import PROJECT_ROOT

config = {
    # Data Path
    "path_data": f"{PROJECT_ROOT}/Dataset/train_split.json", # Initial cohort data
    "val_path_data": f"{PROJECT_ROOT}/Dataset/validation_split.json",

    # Save Path
    "path_save" : f"{PROJECT_ROOT}/logs/output_model/",

    "wandb_run_name": None,  # Optional: set to customize run name
    "enable_wandb": False, # set to True to enable logging

    "random_seed": 42,

    # Data loader
    "patch_size": 12, # 5 mins x 12 = 1 hour

    # Pipeline
    "pipeline": "initial_to_validation", # initial_to_validation, validation_to_validation
    "tag": "Testing reproducibility", # some notes

    # Cross Validation
    "n_splits": 4,

    "task": "classification", # classification

    # Classification
    "extract_method": "ctru_venous", 
    "val_extract_method": "ctru_venous", # ctru_venous, ctru_cgm, home_cgm_1, home_cgm_2, cgm_home_mean, cgm_all_mean 
    "metabolic": "ir", # ir or beta
    "num_class": 1,
    "use_encoder": True,
    "flatten": True, # classical models
    "num_iterations": 1, # iterations to repeat cv

    "train_portion": 1.0,
    "test_portion": 1.0,
    "val_portion": 1.0,

    # Encoder Version
    "cgm_jepa_version": "v9",
    "cgm_jepa_glu_version": "v8", 
    "gluformer_version": "v3",
    "cgm_ts2vec_version": "v0",
    "cgm_mantis_version": "default",  # Set to "default" to enable, None to disable
    "cgm_moment_version": "default",  # Set to "default" to enable, None to disable

    # Optim for epoch based training with decoder
    "num_epochs": 100, 
    "batch_size" : 32,
    "lr": 1e-06,
}
