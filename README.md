# CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining

Official implementation of **CGM-JEPA** and **X-CGM-JEPA** — self-supervised joint embedding-predictive architectures for continuous glucose monitor (CGM) time-series, evaluated on metabolic-subphenotype classification (insulin resistance, β-cell dysfunction).

![CGM-JEPA Overview](/assets/overview.png)

**Key contributions:**
- Masked-patch prediction framework for CGM enabling label-efficient learning.
- Cross-view extension (X-CGM-JEPA) that augments the temporal view with a Glucodensity (distributional) view, improving transfer under distribution shifts.
- Evaluation under cohort generalization, venous-to-CGM modality transfer, and in-domain home CGM.

![X-CGM-JEPA Architecture](/assets/architecture.png)

📄 **Paper:** [arXiv:2605.00933](https://arxiv.org/abs/2605.00933)
🤗 **Pretrained weights:** [`CRUISEResearchGroup/CGM-JEPA`](https://huggingface.co/CRUISEResearchGroup/CGM-JEPA) (model)
🤗 **Downstream splits:** [`CRUISEResearchGroup/CGM-JEPA-Downstream`](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Downstream) (dataset — labeled cohorts for paper Tables 1-6)
🤗 **Pretraining corpus:** [`CRUISEResearchGroup/CGM-JEPA-Pretraining`](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Pretraining) (dataset — raw CGM for SSL pretraining; only needed if you want to re-pretrain)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/cruiseresearchgroup/CGM-JEPA.git
cd CGM-JEPA
conda create -n cgm_jepa python=3.10
conda activate cgm_jepa
pip install -r requirements.txt

# 2. Download the pretrained weights and labeled downstream splits from Hugging Face
huggingface-cli download CRUISEResearchGroup/CGM-JEPA --local-dir Output
huggingface-cli download CRUISEResearchGroup/CGM-JEPA-Downstream --repo-type dataset --local-dir Dataset_Open

# 3. Run the full evaluation (matches paper Tables 1-6, ~15-20 min on Apple Silicon / single GPU)
python scripts/run_all_eval.py
```

If you also want to re-pretrain from scratch, additionally pull the unlabeled corpus:
```bash
huggingface-cli download CRUISEResearchGroup/CGM-JEPA-Pretraining --repo-type dataset --local-dir Dataset_Open
```

---

## Repository structure

```
.
├── config/                  # Pretraining + downstream-eval configurations
│   ├── config_pretrain.py
│   ├── config_downstream.py
│   └── model_configs.py     # Loads pretrained encoders + builds probe lineup
├── data_loaders/            # Dataset loaders (CSV / JSON / patch / token / flat)
├── eval/                    # Downstream evaluation + linear-probe trainer
│   ├── class_reg.py         # Main eval entry point
│   ├── training/
│   └── baseline_utils/      # Mantis, MOMENT, TS2Vec, Ridge baselines
├── models/                  # Architectures: Encoder, Predictor, GluFormer, etc.
├── pretrain/                # Pretraining scripts for each model
├── scripts/
│   ├── preprocess_dataset.py  # Rebuild Dataset_Open/ from upstream sources
│   └── run_all_eval.py        # Orchestrator: 3 settings × 2 endpoints
├── utils/                   # Stats, RevIN, glucodensity precompute
└── README.md
```

After the Hugging Face downloads, you also have:

```
├── Dataset_Open/                      # produced by `huggingface-cli download`
│   ├── train_split.json               #   from CGM-JEPA-Downstream (initial cohort)
│   ├── validation_split.json          #   from CGM-JEPA-Downstream (validation cohort)
│   ├── train.parquet                  #   datasets-library friendly version of train_split
│   ├── validation.parquet             #   datasets-library friendly version of validation_split
│   └── cgm_initial_cohort.csv         #   from CGM-JEPA-Pretraining (only if you re-pretrain)
└── Output/                            # produced by `huggingface-cli download` from CGM-JEPA
    ├── cgm_jepa/
    │   ├── model.safetensors          # PyTorchModelHubMixin format
    │   └── config.json                # architecture hyperparameters
    ├── x_cgm_jepa/
    │   ├── model.safetensors
    │   └── config.json
    └── baselines/
        ├── gluformer.pt
        └── ts2vec.pkl
```

---

## Dataset

Two clinical cohorts (deduplicated; see paper Appendix for full details):

| Cohort | Subjects | Modalities | Lives in HF dataset |
|---|---|---|---|
| **Initial** | 27 | In-clinic venous OGTT | [CGM-JEPA-Downstream](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Downstream) (train split) |
| **Validation** | 17 | In-clinic venous OGTT, in-clinic CGM, two home-CGM windows | [CGM-JEPA-Downstream](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Downstream) (validation split) |
| **Pretraining corpus** | 228 (22 Stanford + 206 Colas) | Continuous home CGM | [CGM-JEPA-Pretraining](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Pretraining) (CSV) |

The two labeled cohorts can also be loaded directly as Hugging Face `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("CRUISEResearchGroup/CGM-JEPA-Downstream")
# DatasetDict({train: Dataset({...,num_rows: 27}), validation: Dataset({...,num_rows: 17})})
```

**Sequence format:** 24-hour windows at 5-min sampling = 288 timesteps = 24 patches of size 12.

**Labels (per subject):**
- `ir`: insulin resistance (binary, derived from SSPG).
- `beta`: β-cell dysfunction (binary, derived from disposition index).

The released dataset is preprocessed (smoothing-spline imputation, label extraction, train/validation deduplication). See "Regenerating the dataset" below if you want to reconstruct it from the original sources.

### Regenerating the dataset

The release on Hugging Face is what we used in the paper. If you want to rebuild it from upstream sources (e.g. for an audit or to apply different preprocessing), [`scripts/preprocess_dataset.py`](scripts/preprocess_dataset.py) is the reference implementation:

```
INPUTS                                              OUTPUTS (Dataset_Open/)
────────────────────────────────────────            ─────────────────────────
Metabolic_Subphenotype_Predictor/data/              train_split.json
  filtered_cgm_03222026.csv                ─┐       validation_split.json
  filtered_ogtt_…_03222026.csv              ├──►    cgm_initial_cohort.csv
  filtered_metabolic_tests.csv              │
                                            │
Dataset/colas_pretrain.parquet            ──┘
```

```bash
# Clone the public Stanford metabolic-subphenotype data source
git clone https://github.com/aametwally/Metabolic_Subphenotype_Predictor.git

# The Colas et al. pretraining corpus is a separate dependency obtained
# from the original authors; place it at Dataset/colas_pretrain.parquet
python scripts/preprocess_dataset.py
```

---

## Pretraining

All commands run from project root. Configurations live in [`config/config_pretrain.py`](config/config_pretrain.py).

**X-CGM-JEPA** — masked CGM-patch prediction + cross-view Glucodensity prediction:
```bash
python -m pretrain.pretrain_x_cgm_jepa
```

**CGM-JEPA** — masked CGM-patch prediction only:
```bash
python -m pretrain.pretrain_cgm_jepa
```

**Baselines** — GluFormer and TS2Vec are also pretrained on our open CGM corpus:
```bash
python -m pretrain.pretrain_gluformer
python -m pretrain.pretrain_ts2vec
```

> **Note on reproducibility from re-pretraining.** Loading the released weights and running the downstream eval reproduces the paper's tables exactly. Re-running pretraining from scratch is expected to land within fold-to-fold noise of the paper's numbers, not bit-exact (due to MPS/CUDA float drift and non-deterministic kernels).

### Required for X-CGM-JEPA — precompute Glucodensity patches

The X-CGM-JEPA cross-view objective needs Glucodensity images. Run this once after downloading the pretraining corpus (a few minutes on CPU):

```bash
python -m utils.precompute_glucodensity \
  --data_path Dataset_Open/cgm_initial_cohort.csv \
  --output_path Dataset_Open/gluco_cache.pkl \
  --stride 144
```

The cache (~54 MB) is **not** shipped on Hugging Face — it's derived deterministically from the CSV, so we leave the regeneration step to users to keep the dataset repo lean.

---

## Loading models programmatically

Both pretrained CGM-JEPA encoders subclass `huggingface_hub.PyTorchModelHubMixin`, so a single `from_pretrained` call downloads weights + architecture hyperparameters and instantiates the model:

```python
from models.encoder import Encoder

cgm_jepa   = Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA", subfolder="cgm_jepa")
x_cgm_jepa = Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA", subfolder="x_cgm_jepa")
```

The `config.json` packaged alongside each `model.safetensors` carries `patch_size`, `embed_dim`, `nhead`, `num_layers`, etc. — no manual hyperparameter wiring is needed.

For the GluFormer / TS2Vec baselines (stored under `Output/baselines/` after the download), see the model card on Hugging Face for the standalone-PyTorch loading example.

---

## Evaluation

We evaluate two binary outcomes (insulin resistance, β-cell dysfunction) under three regimes (each table corresponds to a `(pipeline, extract_method, val_extract_method)` config triple):

| Regime | Config |
|---|---|
| **In-domain home CGM** | `validation_to_validation`, `cgm_home_mean → cgm_home_mean` |
| **Venous → home-CGM transfer** | `validation_to_validation`, `ctru_venous → cgm_home_mean` |
| **Cohort generalization (venous)** | `initial_to_validation`, `ctru_venous → ctru_venous` |

Each cell is 20 iterations × 2-fold stratified CV (40 paired observations per model).

**Run everything (3 settings × 2 endpoints):**
```bash
python scripts/run_all_eval.py
```

**Run one cell directly:**
```bash
python -m eval.class_reg
```
(Configure pipeline / metabolic / extract_method in [`config/config_downstream.py`](config/config_downstream.py).)

**Evaluated models:** CGM-JEPA, X-CGM-JEPA, GluFormer, TS2Vec, MOMENT-Small, MOMENT-Large, Mantis, plus a PCA + L2-LR baseline.

**Metrics:** AUROC, F1, PRAUC (downstream); Silhouette, Calinski–Harabasz, Davies–Bouldin, ARI, NMI, between/within ratio, intra/inter-cluster distance (representation).

---

## Citation

```bibtex
@article{muhammad2026cgm,
  title   = {CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining},
  author  = {Muhammad, Hada Melino and Li, Zechen and Salim, Flora and Metwally, Ahmed A},
  journal = {arXiv preprint arXiv:2605.00933},
  year    = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)).
