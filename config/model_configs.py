import copy
import os
import torch

import torch.nn as nn

from root import PROJECT_ROOT

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from eval.baseline_utils.ridge_utils import AdaptiveCalibratedRidgeClassifier

from .config_downstream import config

from models.encoder import Encoder
from models.gluformer.gluformer import GluFormer

from utils.main_utils import load_device


# Local layout under Output/ as published on Hugging Face
# (https://huggingface.co/CRUISEResearchGroup/CGM-JEPA). The JEPA encoders use
# the PyTorchModelHubMixin layout (directory with model.safetensors + config.json),
# while the baselines retain their original on-disk formats.
_LOCAL_WEIGHTS = {
    "cgm_jepa": {
        "path": "Output/cgm_jepa",          # directory for Encoder.from_pretrained
        "metadata": {
            "patch_size": 12,
            "encoder_kernel_size": 3,
            "encoder_embed_dim": 96,
            "encoder_embed_bias": True,
            "encoder_nhead": 6,
            "encoder_num_layers": 3,
        },
    },
    "x_cgm_jepa": {
        "path": "Output/x_cgm_jepa",        # directory
        "metadata": {
            "patch_size": 12,
            "encoder_kernel_size": 3,
            "encoder_embed_dim": 96,
            "encoder_embed_bias": True,
            "encoder_nhead": 6,
            "encoder_num_layers": 3,
        },
    },
    "gluformer": {
        "path": "Output/baselines/gluformer.pt",  # file
        "metadata": {
            "vocab_size": 278,
            "embed_dim": 96,
            "nhead": 6,
            "num_layers": 3,
        },
    },
    "ts2vec": {
        "path": "Output/baselines/ts2vec.pkl",    # file
        "metadata": {
            "output_dims": 96,
            "hidden_dims": 64,
            "depth": 10,
        },
    },
}


def _resolve_local(name):
    """Return (absolute_path, metadata) for the local checkpoint.

    Raises if the asset isn't present, pointing the user to the HF download.
    """
    path = os.path.join(PROJECT_ROOT, _LOCAL_WEIGHTS[name]["path"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Local weight asset for '{name}' not found at {path}.\n"
            f"Run:\n"
            f"  huggingface-cli download CRUISEResearchGroup/CGM-JEPA --local-dir Output\n"
            f"(see https://huggingface.co/CRUISEResearchGroup/CGM-JEPA)."
        )
    return path, _LOCAL_WEIGHTS[name]["metadata"]

CLASSIFIER_SPECS = [
    {
        "key": "L2_LR",
        "name_suffix": "L2_LR",
        "factory": lambda seed, _C: LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            cv=2,
            scoring="roc_auc",
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
    seed, LR_C, LSVC_C, R_alpha, RF_n_estimators, KNN_n_neighbors, exclude_knn=False, classifier_specs=None
):
    """Baselines without encoders."""
    specs = classifier_specs or CLASSIFIER_SPECS
    configs = []
    for spec in specs:
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
    classifier_specs=None,
):
    """Shared factory for encoder-based probe configs."""
    specs = classifier_specs or CLASSIFIER_SPECS
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
    for spec in specs:
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
    linear_probe_only = config.get("linear_probe_only", False)

    # Filter classifier specs if linear_probe_only
    active_specs = [s for s in CLASSIFIER_SPECS if s["key"] == "L2_LR"] if linear_probe_only else CLASSIFIER_SPECS

    # ---- Resolve checkpoints from Hugging Face ----
    # Encoders are loaded directly from
    #   https://huggingface.co/CRUISEResearchGroup/CGM-JEPA
    # which the project's Quick start tells external users to download into
    # `Output/` via `huggingface-cli download CRUISEResearchGroup/CGM-JEPA --local-dir Output`.
    # If you need the wandb-backed loader (artifact versions pinned to
    # cgm-jepa:v18, cgm-jepa-glucodensity-separate:v19, gluformer:v5, ts2vec:v2),
    # use `config/model_configs_wandb.py` (gitignored, personal use only).

    def _load_encoder_from_hf(name):
        path, meta = _resolve_local(name)
        return Encoder.from_pretrained(path).to(device), meta

    cgm_jepa,   cgm_jepa_metadata   = _load_encoder_from_hf("cgm_jepa")
    x_cgm_jepa, x_cgm_jepa_metadata = _load_encoder_from_hf("x_cgm_jepa")

    # GluFormer (legacy state_dict .pt; PyTorchModelHubMixin not used)
    gluformer_ckpt, gluformer_metadata = _resolve_local("gluformer")
    gf_embed_dim  = gluformer_metadata["embed_dim"]
    gf_vocab_size = gluformer_metadata.get("vocab_size", 280)
    gluformer = GluFormer(
        vocab_size=gf_vocab_size,
        embed_dim=gf_embed_dim,
        nhead=gluformer_metadata["nhead"],
        num_layers=gluformer_metadata["num_layers"],
        dim_feedforward=2 * gf_embed_dim,
        max_seq_length=25000,
        dropout=0.0,
        pad_token=gf_vocab_size,
    ).to(device)
    gluformer.load_state_dict(
        torch.load(gluformer_ckpt, map_location=device)["encoder"]
    )
    gluformer.output_head = nn.Identity()

    # # classification
    model_configs = make_baseline_configs(
        seed, LR_C, LSVC_C, R_alpha, RF_n_estimators, KNN_n_neighbors,
        exclude_knn=exclude_knn, classifier_specs=active_specs,
    )

    # Untrained encoders (random initialization baseline)
    untrained_jepa = Encoder(
        dim_in=cgm_jepa_metadata["patch_size"],
        kernel_size=cgm_jepa_metadata["encoder_kernel_size"],
        embed_dim=cgm_jepa_metadata["encoder_embed_dim"],
        embed_bias=cgm_jepa_metadata["encoder_embed_bias"],
        nhead=cgm_jepa_metadata["encoder_nhead"],
        num_layers=cgm_jepa_metadata["encoder_num_layers"],
        jepa=False
    ).to(device)

    untrained_x_jepa = Encoder(
        dim_in=x_cgm_jepa_metadata["patch_size"],
        kernel_size=x_cgm_jepa_metadata["encoder_kernel_size"],
        embed_dim=x_cgm_jepa_metadata["encoder_embed_dim"],
        embed_bias=x_cgm_jepa_metadata["encoder_embed_bias"],
        nhead=x_cgm_jepa_metadata["encoder_nhead"],
        num_layers=x_cgm_jepa_metadata["encoder_num_layers"],
        jepa=False
    ).to(device)

    # Probes with trained encoders
    # CGM-JEPA
    model_configs.extend(make_probe_configs(
        seed, encoder=cgm_jepa, prefix="Pretrained_JEPA_Encoder",
        output_type="patch", patchify=True,
        LR_C=LR_C, LSVC_C=LSVC_C, R_alpha=R_alpha,
        RF_n_estimators=RF_n_estimators, KNN_n_neighbors=KNN_n_neighbors,
        include_knn=True, exclude_knn=exclude_knn, classifier_specs=active_specs,
    ))
    # X-CGM-JEPA
    model_configs.extend(make_probe_configs(
        seed, encoder=x_cgm_jepa, prefix="Pretrained_JEPA_Glu_Encoder",
        output_type="patch", patchify=True,
        LR_C=LR_C, LSVC_C=LSVC_C, R_alpha=R_alpha,
        RF_n_estimators=RF_n_estimators, KNN_n_neighbors=KNN_n_neighbors,
        include_knn=True, exclude_knn=exclude_knn, classifier_specs=active_specs,
    ))
    # Untrained CGM-JEPA
    model_configs.extend(make_probe_configs(
        seed, encoder=untrained_jepa, prefix="Untrained_JEPA_Encoder",
        output_type="patch", patchify=True,
        LR_C=LR_C, LSVC_C=LSVC_C, R_alpha=R_alpha,
        RF_n_estimators=RF_n_estimators, KNN_n_neighbors=KNN_n_neighbors,
        include_knn=True, exclude_knn=exclude_knn, classifier_specs=active_specs,
    ))
    # Untrained X-CGM-JEPA
    model_configs.extend(make_probe_configs(
        seed, encoder=untrained_x_jepa, prefix="Untrained_JEPA_Glu_Encoder",
        output_type="patch", patchify=True,
        LR_C=LR_C, LSVC_C=LSVC_C, R_alpha=R_alpha,
        RF_n_estimators=RF_n_estimators, KNN_n_neighbors=KNN_n_neighbors,
        include_knn=True, exclude_knn=exclude_knn, classifier_specs=active_specs,
    ))
    # GluFormer
    model_configs.extend(make_probe_configs(
        seed, encoder=gluformer, prefix="Pretrained_GluFormer",
        output_type="token", patchify=False,
        LR_C=LR_C, LSVC_C=LSVC_C, R_alpha=R_alpha,
        RF_n_estimators=RF_n_estimators, KNN_n_neighbors=KNN_n_neighbors,
        include_knn=True, exclude_knn=exclude_knn, classifier_specs=active_specs,
    ))

    # TS2Vec (pickled full-model object; PyTorchModelHubMixin not applicable)
    from eval.baseline_utils.ts2vec_utils import load_pretrained_ts2vec, TS2VecEncoderWrapper
    checkpoint_path, ts2vec_meta = _resolve_local("ts2vec")
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
            classifier_specs=active_specs,
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
                    "normalize_for_encoder": False,
                },
                classifier_specs=active_specs,
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
                classifier_specs=active_specs,
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
                classifier_specs=active_specs,
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