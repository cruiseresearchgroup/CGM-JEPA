import copy
import os
import torch
import wandb

import torch.nn as nn

from root import PROJECT_ROOT

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from eval.baseline_utils.ridge_utils import AdaptiveCalibratedRidgeClassifier

from .config_downstream import config

from models.encoder import Encoder
from models.gluformer.gluformer import GluFormer

from utils.main_utils import load_device

CLASSIFIER_SPECS = [
    {
        "key": "L2_LR",
        "name_suffix": "L2_LR",
        "factory": lambda seed, C: LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    },
    {
        "key": "Linear_SVC",
        "name_suffix": "Linear_SVC",
        "factory": lambda seed, C: SVC(
            kernel="linear",
            C=C,
            probability=True,
            random_state=seed,
            class_weight="balanced",
        ),
    },
    {
        "key": "Ridge",
        "name_suffix": "Ridge",
        "factory": lambda seed, alpha: AdaptiveCalibratedRidgeClassifier(
            alpha=alpha,
            random_state=seed,
            class_weight="balanced",
        ),
    },
    {
        "key": "RandomForest",
        "name_suffix": "RandomForest",
        "factory": lambda seed, n_estimators: RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            class_weight="balanced",
        ),
    },
    {
        "key": "KNN",
        "name_suffix": "KNN",
        "factory": lambda seed, n_neighbors: KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
        ),
    },
]


def _build_classifier_from_spec(
    spec, seed, LR_C, LSVC_C, R_alpha, RF_n_estimators, KNN_n_neighbors
):
    key = spec["key"]
    if key == "L2_LR":
        return spec["factory"](seed, LR_C)
    if key == "Linear_SVC":
        return spec["factory"](seed, LSVC_C)
    if key == "Ridge":
        return spec["factory"](seed, R_alpha)
    if key == "RandomForest":
        return spec["factory"](seed, RF_n_estimators)
    if key == "KNN":
        return spec["factory"](seed, KNN_n_neighbors)
    raise ValueError(f"Unknown classifier key: {key}")


def make_baseline_configs(
    seed, LR_C, LSVC_C, R_alpha, RF_n_estimators, KNN_n_neighbors, exclude_knn=False
):
    """Baselines without encoders."""
    configs = []
    for spec in CLASSIFIER_SPECS:
        if spec["key"] == "KNN" and exclude_knn:
            continue

        classifier = _build_classifier_from_spec(
            spec, seed, LR_C, LSVC_C, R_alpha, RF_n_estimators, KNN_n_neighbors
        )

        configs.append(
            {
                "name": spec["name_suffix"],
                "classifier": classifier,
                "train_type": "classical",
                "output_type": "flat",
            }
        )
    return configs


def make_probe_configs(
    seed,
    encoder,
    prefix,
    output_type,
    patchify,
    LR_C,
    LSVC_C,
    R_alpha,
    RF_n_estimators,
    KNN_n_neighbors,
    include_knn=True,
    exclude_knn=False,
    extra_common=None,
):
    """Shared factory for encoder-based probe configs."""
    common = {
        "encoder": encoder,
        "use_encoder": True,
        "patchify": patchify,
        "train_type": "classical",
        "output_type": output_type,
    }
    if extra_common:
        common.update(extra_common)

    configs = []
    for spec in CLASSIFIER_SPECS:
        if spec["key"] == "KNN" and (exclude_knn or not include_knn):
            continue

        classifier = _build_classifier_from_spec(
            spec, seed, LR_C, LSVC_C, R_alpha, RF_n_estimators, KNN_n_neighbors
        )

        configs.append(
            {
                "name": f"{prefix}_{spec['name_suffix']}_Probe",
                "classifier": classifier,
                **common,
            }
        )
    return configs


def get_model_configs(config):
    device = load_device()

    LR_C = 0.1
    LSVC_C = 10
    R_alpha = 10
    RF_n_estimators = 100
    KNN_n_neighbors = 5

    seed = config["random_seed"]
    exclude_knn = config.get("exclude_knn", False)

    cgm_jepa_version = config["cgm_jepa_version"]
    gluformer_version = config["gluformer_version"]
    cgm_jepa_glu_version = config["cgm_jepa_glu_version"]
    cgm_ts2vec_version = config["cgm_ts2vec_version"]

    api = wandb.Api()

    # CGM-JEPA
    cgm_jepa_artifact = api.artifact(f'hadamuhammad-unsw/cgm-jepa/cgm-jepa:{cgm_jepa_version}', type="model")
    cgm_jepa_metadata = cgm_jepa_artifact.metadata

    cgm_jepa = Encoder(
        dim_in=cgm_jepa_metadata["patch_size"],
        kernel_size=cgm_jepa_metadata["encoder_kernel_size"],
        embed_dim=cgm_jepa_metadata["encoder_embed_dim"],
        embed_bias=cgm_jepa_metadata["encoder_embed_bias"],
        nhead=cgm_jepa_metadata["encoder_nhead"],
        num_layers=cgm_jepa_metadata["encoder_num_layers"],
        jepa=False # we don't apply jepa training in downstream
    ).to(device)
    cgm_jepa_dir = cgm_jepa_artifact.download()
    cgm_jepa.load_state_dict(torch.load(f"{cgm_jepa_dir}/cgm-jepa", map_location=device)["encoder"], strict=False)
    
    # X-CGM-JEPA
    x_cgm_jepa_artifact = api.artifact(f'hadamuhammad-unsw/cgm-jepa-glucodensity-separate/cgm-jepa-glucodensity-separate:{cgm_jepa_glu_version}', type="model")
    x_cgm_jepa_dir = x_cgm_jepa_artifact.download()
    x_cgm_jepa_metadata = x_cgm_jepa_artifact.metadata
    x_cgm_jepa = Encoder(
        dim_in=x_cgm_jepa_metadata["patch_size"], # patch size
        kernel_size=x_cgm_jepa_metadata["encoder_kernel_size"],
        embed_dim=x_cgm_jepa_metadata["encoder_embed_dim"],
        embed_bias=x_cgm_jepa_metadata["encoder_embed_bias"],
        nhead=x_cgm_jepa_metadata["encoder_nhead"],
        num_layers=x_cgm_jepa_metadata["encoder_num_layers"],
        jepa=False # we don't apply jepa training in downstream
    ).to(device)
    x_cgm_jepa.load_state_dict(torch.load(f"{x_cgm_jepa_dir}/lr_0.0001_ema_momentum_0.997_mask_ratio_{x_cgm_jepa_metadata['mask_ratio']}_encoder_96_6_3_predictor_48_2_1_not_use_time.pt", map_location=device)["cgm_encoder"], strict=True)

    # GluFormer
    gluformer_artifact = api.artifact(f'hadamuhammad-unsw/gluformer/gluformer:{gluformer_version}', type="model")
    gluformer_metadata = gluformer_artifact.metadata
    gluformer = GluFormer(
        vocab_size=280,
        embed_dim=gluformer_metadata["encoder_embed_dim"],
        nhead=gluformer_metadata["encoder_nhead"],
        num_layers=gluformer_metadata["encoder_num_layers"],
        dim_feedforward=2 * gluformer_metadata["encoder_embed_dim"],
        max_seq_length=25000,
        dropout=0.0,
        pad_token=280
    ).to(device)
    gluformer_dir = gluformer_artifact.download()
    gluformer.load_state_dict(torch.load(f"{gluformer_dir}/gluformer", map_location=device)["encoder"])
    # detach GluFormer head
    gluformer.output_head = nn.Identity()

    # # classification
    model_configs = make_baseline_configs(
        seed,
        LR_C,
        LSVC_C,
        R_alpha,
        RF_n_estimators,
        KNN_n_neighbors,
        exclude_knn=exclude_knn,
    )

    # Probes with trained encoders
    # CGM-JEPA
    model_configs.extend(
        make_probe_configs(
            seed,
            encoder=cgm_jepa,
            prefix="Pretrained_JEPA_Encoder",
            output_type="patch",
            patchify=True,
            LR_C=LR_C,
            LSVC_C=LSVC_C,
            R_alpha=R_alpha,
            RF_n_estimators=RF_n_estimators,
            KNN_n_neighbors=KNN_n_neighbors,
            include_knn=True,
            exclude_knn=exclude_knn,
        )
    )
    # X-CGM-JEPA
    model_configs.extend(
        make_probe_configs(
            seed,
            encoder=x_cgm_jepa,
            prefix="Pretrained_JEPA_Glu_Encoder",
            output_type="patch",
            patchify=True,
            LR_C=LR_C,
            LSVC_C=LSVC_C,
            R_alpha=R_alpha,
            RF_n_estimators=RF_n_estimators,
            KNN_n_neighbors=KNN_n_neighbors,
            include_knn=True,
            exclude_knn=exclude_knn,
        )
    )
    # GluFormer
    model_configs.extend(
        make_probe_configs(
            seed,
            encoder=gluformer,
            prefix="Pretrained_GluFormer",
            output_type="token",
            patchify=False,
            LR_C=LR_C,
            LSVC_C=LSVC_C,
            R_alpha=R_alpha,
            RF_n_estimators=RF_n_estimators,
            KNN_n_neighbors=KNN_n_neighbors,
            include_knn=True,
            exclude_knn=exclude_knn,
        )
    )

    # TS2Vec (added more baselines)
    from eval.baseline_utils.ts2vec_utils import load_pretrained_ts2vec, TS2VecEncoderWrapper
    ts2vec_artifact = api.artifact(
        f"hadamuhammad-unsw/cgm-ts2vec-pretrain/cgm-ts2vec:{cgm_ts2vec_version}",
        type="model",
    )
    ts2vec_dir = ts2vec_artifact.download()
    checkpoint_path = os.path.join(ts2vec_dir, "cgm-ts2vec.pkl")
    ts2vec_meta = ts2vec_artifact.metadata or {}
    output_dims = ts2vec_meta.get("output_dims", config.get("ts2vec_output_dims", 96))
    hidden_dims = ts2vec_meta.get("hidden_dims", config.get("ts2vec_hidden_dims", 64))
    depth = ts2vec_meta.get("depth", config.get("ts2vec_depth", 10))
    ts2vec_model = load_pretrained_ts2vec(
        checkpoint_path=checkpoint_path,
        device="cpu",
        input_dims=1,
        output_dims=output_dims,
        hidden_dims=hidden_dims,
        depth=depth,
    )
    ts2vec_encoder = TS2VecEncoderWrapper(ts2vec_model, "cpu")
    ts2vec_encoder = ts2vec_encoder.to(device) 
    model_configs.extend(
        make_probe_configs(
            seed,
            encoder=ts2vec_encoder,
            prefix="Pretrained_TS2Vec_Encoder",
            output_type="flat",
            patchify=False,
            LR_C=LR_C,
            LSVC_C=LSVC_C,
            R_alpha=R_alpha,
            RF_n_estimators=RF_n_estimators,
            KNN_n_neighbors=KNN_n_neighbors,
            include_knn=True,
            exclude_knn=exclude_knn,
        )
    )

    # Mantis (added more baselines)
    cgm_mantis_version = config.get("cgm_mantis_version")
    if cgm_mantis_version:
        from eval.baseline_utils.mantis_utils import load_mantis_encoder
        mantis_encoder = load_mantis_encoder(device)
        model_configs.extend(
            make_probe_configs(
                seed,
                encoder=mantis_encoder,
                prefix="Pretrained_Mantis_Encoder",
                output_type="flat",
                patchify=False,
                LR_C=LR_C,
                LSVC_C=LSVC_C,
                R_alpha=R_alpha,
                RF_n_estimators=RF_n_estimators,
                KNN_n_neighbors=KNN_n_neighbors,
                include_knn=False,
                exclude_knn=exclude_knn,
                extra_common={
                    "normalize_for_encoder": False,  # Mantis expects raw (unnormalized) time series
                },
            )
        )

    # MOMENT (added more baselines)
    cgm_moment_version = config.get("cgm_moment_version")
    if cgm_moment_version:
        from eval.baseline_utils.moment_utils import load_moment_encoder
        moment_encoder_large = load_moment_encoder(device, model_name="AutonLab/MOMENT-1-large")
        moment_encoder_small = load_moment_encoder(device, model_name="AutonLab/MOMENT-1-small")
        moment_common_extra = {
            "normalize_for_encoder": False,  # MOMENT expects raw (unnormalized) time series
        }
        # Large
        model_configs.extend(
            make_probe_configs(
                seed,
                encoder=moment_encoder_large,
                prefix="MOMENT_Large_Encoder",
                output_type="flat",
                patchify=False,
                LR_C=LR_C,
                LSVC_C=LSVC_C,
                R_alpha=R_alpha,
                RF_n_estimators=RF_n_estimators,
                KNN_n_neighbors=KNN_n_neighbors,
                include_knn=False,
                exclude_knn=exclude_knn,
                extra_common=moment_common_extra,
            )
        )
        # Small
        model_configs.extend(
            make_probe_configs(
                seed,
                encoder=moment_encoder_small,
                prefix="MOMENT_Small_Encoder",
                output_type="flat",
                patchify=False,
                LR_C=LR_C,
                LSVC_C=LSVC_C,
                R_alpha=R_alpha,
                RF_n_estimators=RF_n_estimators,
                KNN_n_neighbors=KNN_n_neighbors,
                include_knn=False,
                exclude_knn=exclude_knn,
                extra_common=moment_common_extra,
            )
        )

    # Restrict to configs whose name contains any of the given substrings (e.g. only JEPA_Glu for ablation)
    only_encoders = config.get("only_encoders")
    if only_encoders:
        model_configs = [
            c for c in model_configs
            if any(sub in c.get("name", "") for sub in only_encoders)
        ]

    return model_configs