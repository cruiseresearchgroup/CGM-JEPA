# CGM-JEPA: Learning Transferable Continuous Glucose Monitor Representations via Self-Supervised Pretraining and Cross-View Regularization

Official implementation of **CGM-JEPA** and **X-CGM-JEPA** - self-supervised joint embedding prediction architectures for continuous glucose monitoring time-series data.

![CGM-JEPA Overview](/assets/overview.png)

**Key contributions:**
- Masked prediction framework for CGM patches enabling label-efficient learning
- Cross-view extension (X-CGM-JEPA) for improved transfer under distribution shifts
- Evaluation under cohort generalization, venous-to-CGM transfer, and real-world deployment

![X-CGM-JEPA Architecture](/assets/architecture.png)

📄 **Paper:** [Link TBD]

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/cruiseresearchgroup/CGM-JEPA.git
cd cgm_jepa

conda create -n cgm_jepa python=3.10
conda activate cgm_jepa
pip install -r requirements.txt

# Pretrain X-CGM-JEPA
python -m pretrain.pretrain_x_cgm_jepa

# Evaluate on downstream tasks
python -m eval.class_reg
```

---

## Dataset

Two clinical cohorts with complementary modality availability:

| Cohort | Size | Data Type | Usage |
|--------|------|-----------|-------|
| Initial | N=32 | Unlabeled home CGM + labeled in-clinic venous glucose | Pretraining + evaluation |
| Validation | N=24 | Labeled in-clinic venous glucose + in-clinic/home CGM | Evaluation |

**Data format:** 24-hour sequences represented as **24 hourly patches** (12 points/hour at 5-min sampling = 288 points/day)

### Expected CSV format
```
subject,timestamp,glucose_value
001,2024-01-01 00:00:00,95
001,2024-01-01 00:05:00,97
...
```

- 5-minute sampling interval
- Place data files in `Dataset/` directory

---

## Pretraining

All commands run from project root. Configurations in `config/config_pretrain.py`.

### X-CGM-JEPA
```bash
python -m pretrain.pretrain_x_cgm_jepa
```
Masked prediction on CGM patches + cross-view glucodensity objective. Best for transfer scenarios.

### CGM-JEPA
```bash
python -m pretrain.pretrain_cgm_jepa
```
Masked representation prediction on CGM patches only. Strong in-domain performance.

### Baseline models

**GluFormer:**
```bash
python -m pretrain.pretrain_gluformer
```

**TS2Vec:**
```bash
python -m pretrain.pretrain_ts2vec
```

### Optional: Precompute glucodensity patches

For faster training:

```bash
python -m utils.precompute_glucodensity \
  --data_path Dataset/cgm_initial_cohort.csv \
  --output_path Dataset/gluco_cache.pkl
```

Update `config["gluco_cache_path"]` in `config/config_pretrain.py` to use cached patches.

---

## Evaluation

Evaluate two binary outcomes (insulin resistance and β-cell dysfunction) under three settings:

1. **Cohort generalization:** Train on initial cohort, test on validation cohort
2. **Venous-to-CGM transfer:** Train with venous supervision, test on CGM embeddings
3. **Real-world home CGM:** Train and test on home CGM

```bash
python -m eval.class_reg
```

**Configuration:** `config/config_downstream.py`

**Evaluated models:**
- **Pretrained encoders:** CGM-JEPA, X-CGM-JEPA, GluFormer, TS2Vec, MOMENT, Mantis (linear probe)
- **Classical baselines:** Logistic regression, linear SVM, ridge regression, random forest, KNN

**Metrics:** AUROC, AUPRC, F1, representation quality metrics

## Reproducibility

- **Seed control:** Set random seeds in config files
- **Weights & Biases:** Enable with `enable_wandb=True` in configs for experiment tracking
- **Hyperparameters:** All hyperparameters documented in `config/` files
