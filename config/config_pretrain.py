from root import PROJECT_ROOT

config = {    

    # Data used for pretraining (initial cohort only)
    "path_data": f"{PROJECT_ROOT}/Dataset/cgm_initial_cohort.csv",

    # Base path to save the output
    "path_save": f"{PROJECT_ROOT}/Output",

    "enable_wandb" : True,
    "batch_size" : 128,
    "random_seed": 43,
    
    # Loader
    "mask_ratio" : 0.75,
    "gluco_loss_weight": 1.0,  # Weight for glucodensity loss (increase if gluco loss is too high)
    "patch_size": 12,

    "clip_grad_max_norm": 1.0,
    "warmup_ratio": 0.15,
    "ipe_scale": 1.25,

    # NOTE: Only for experiments adding time feature to the embedding
    # since the evaluation dataset doesn't have timestamp, we don't use this
    # to prevent embedding mismatch
    "use_time_feature": False, 
    # Time embedding (Informer style)
    "time_inp_dim": 5,

    #optim
    "lr": 1e-4,
    "gamma": 0.99,
    "step_size": 100,
    "num_epochs": 101,
    "ema_momentum" : 0.997,

    # Encoder
    "encoder_embed_dim" : 96,
    "encoder_nhead" : 6,
    "encoder_num_layers": 3,
    "encoder_kernel_size" : 3,
    "encoder_embed_bias": True,
    "encoder_dropout": 0.0,

    # Predictor
    "predictor_embed" : 48,
    "predictor_nhead" : 2,
    "predictor_num_layers": 1,

    # Glucodensity KDE computation
    "gluco_kde_workers": 8,  # Number of parallel workers for KDE computation (CPU-bound)
    "gluco_spatial_patch_size": 8,  # Spatial patch size for glucodensity (8x8)
    "gluco_gridsize": 32,  # KDE grid size (32x32)
    
    # Run pre-computation first: python -m utils.precompute_glucodensity --data_path <path> --output_path <cache_path>
    "gluco_cache_path": "",  # Path to pre-computed glucodensity patches (set to speed up training!)
                            
    # TS2Vec pretraining configuration
    "ts2vec_output_dims": 96,  # Representation dimension
    "ts2vec_hidden_dims": 64,  # Hidden dimension of encoder
    "ts2vec_depth": 10,  # Number of hidden residual blocks
    "ts2vec_lr": 0.001,  # Learning rate
    "ts2vec_batch_size": 128,  # Batch size
    "ts2vec_max_train_length": 3000,  # Maximum sequence length for training
    "ts2vec_n_epochs": 100,  # Number of epochs
    "ts2vec_n_iters": None,  # Number of iterations (if None, uses n_epochs)
    }
