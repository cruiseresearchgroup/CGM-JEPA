"""
Generate open-source version of our Dataset using the Metabolic_Subphenotype_Predictor repo data.

Produces Dataset_Open/:
    - train_split.json       — initial cohort (ctru_venous only, smoothed to 5-min grid)
    - validation_split.json  — validation cohort (ctru_venous, ctru_cgm, home_cgm_1, home_cgm_2 + derived means)
    - cgm_initial_cohort.csv — continuous CGM for SSL pretraining
                               (initial cohort subjects + colas_pretrain.parquet daily windows)

Follows the same preprocessing logic as notebooks/exp.ipynb.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.interpolate import make_smoothing_spline

BASE = Path(__file__).parent.parent
MSP_DATA = BASE / "Metabolic_Subphenotype_Predictor" / "data"
OUT = BASE / "Dataset_Open"
OUT.mkdir(exist_ok=True)

# ---------- smoothing helper ----------
def apply_smoothing(x_raw, lam=0.35):
    x_arr = np.array(x_raw, dtype=float)
    time_indices = np.arange(len(x_raw))
    valid_mask = x_arr != -1
    if valid_mask.sum() < 4:  # need enough points for spline
        return x_arr.tolist()
    valid_x = x_arr[valid_mask]
    valid_indices = time_indices[valid_mask]
    spline = make_smoothing_spline(x=valid_indices, y=valid_x, lam=lam)
    x_arr = spline(time_indices)
    return x_arr.tolist()


# ---------- load MSP data ----------
print("Loading MSP repo data...")
cgm = pd.read_csv(MSP_DATA / "filtered_cgm_03222026.csv")
# Handle "Low"/"High" markers: replace with numeric min/max of valid values
_numeric = pd.to_numeric(cgm["glucose_value"], errors="coerce")
_min, _max = int(_numeric.min()), int(_numeric.max())
cgm.loc[cgm["glucose_value"] == "Low", "glucose_value"] = _min
cgm.loc[cgm["glucose_value"] == "High", "glucose_value"] = _max
cgm["glucose_value"] = pd.to_numeric(cgm["glucose_value"])
print(f"  Cleaned Stanford CGM: Low -> {_min}, High -> {_max}")
ogtt = pd.read_csv(MSP_DATA / "filtered_ogtt_glucose_timeseries_ctru_athome_venous_cgm_03222026.csv")
meta = pd.read_csv(MSP_DATA / "filtered_metabolic_tests.csv")

# Rename for compatibility with exp.ipynb logic
meta = meta.rename(columns={"SubjectID": "subject_id", "exp_type": "exp_type"})

INITIAL_EXP = "venous_without_matching_cgm_and_without_planned_athome_cgm"
VALIDATION_EXP = "venous_with_matching_cgm_and_with_planned_athome_cgm"

# ---------- subject cohorts ----------
initial_subjects = set(meta[meta["exp_type"] == INITIAL_EXP]["subject_id"].unique())
validation_subjects_raw = set(meta[meta["exp_type"] == VALIDATION_EXP]["subject_id"].unique())

# Remove overlapping subjects from validation (exp.ipynb cell 20 logic)
validation_subjects = sorted(validation_subjects_raw - initial_subjects)
initial_subjects = sorted(initial_subjects)

print(f"Initial cohort: {len(initial_subjects)} subjects")
print(f"Validation cohort (deduplicated): {len(validation_subjects)} subjects")
print(f"  Removed {len(validation_subjects_raw) - len(validation_subjects)} subjects that also appear in initial")


# ---------- helper: label extraction ----------
def get_labels(meta_row):
    sspg = meta_row["sspg_2_classes"]
    di = meta_row["di_2_classes_median"]
    sspg_val = meta_row["sspg"]
    di_val = meta_row["di"]
    return {
        "ir": {
            "class": 0 if sspg == "IS" else 1 if sspg == "IR" else -1,
            "regression": float(sspg_val) if pd.notna(sspg_val) and str(sspg_val) != "NA" else -1,
        },
        "beta": {
            "class": 0 if di == "Normal" else 1 if di == "Dysfunction" else -1,
            "regression": float(di_val) if pd.notna(di_val) and str(di_val) != "NA" else -1,
        },
    }


def align_timeseries(x_df, t_min=-10, t_max=180, step=5):
    """Align OGTT timeseries to 5-min grid, filling missing with -1."""
    x_align = []
    t = t_min
    while t <= t_max:
        g = x_df[x_df["Timepoint"] == t]["Glucose"]
        x_align.append(int(g.iloc[0]) if not g.empty else -1)
        t += step
    return x_align


# ---------- train_split (initial cohort) ----------
print("\nGenerating train_split.json...")
train_split = {}
for s in initial_subjects:
    meta_rows = meta[(meta["subject_id"] == s) & (meta["exp_type"] == INITIAL_EXP)]
    if meta_rows.empty:
        continue
    meta_row = meta_rows.iloc[0]

    x_df = (
        ogtt[(ogtt["SubjectID"] == s)
             & (ogtt["SampleLocation_ExtractionMethod"] == "CTRU_Venous")
             & (ogtt["ExperimentType"] == INITIAL_EXP)]
        [["Timepoint", "Glucose"]]
        .sort_values("Timepoint")
        .reset_index(drop=True)
    )
    if x_df.empty:
        print(f"  Warning: no CTRU_Venous for {s}, skipping")
        continue

    x_align = align_timeseries(x_df)
    x_align = apply_smoothing(x_align, lam=0.35)

    train_split[s] = {
        "x": {"ctru_venous": x_align},
        "y": get_labels(meta_row),
    }

with open(OUT / "train_split.json", "w") as f:
    json.dump(train_split, f, indent=4)
print(f"  Saved {len(train_split)} subjects to {OUT/'train_split.json'}")


# ---------- validation_split ----------
print("\nGenerating validation_split.json...")
extract_methods = ogtt["SampleLocation_ExtractionMethod"].unique().tolist()

validation_split = {}
for s in validation_subjects:
    meta_rows = meta[(meta["subject_id"] == s) & (meta["exp_type"] == VALIDATION_EXP)]
    if meta_rows.empty:
        continue
    meta_row = meta_rows.iloc[0]

    subj_entry = {"x": {}, "y": get_labels(meta_row)}

    for em in extract_methods:
        x_df = (
            ogtt[(ogtt["SubjectID"] == s)
                 & (ogtt["SampleLocation_ExtractionMethod"] == em)]
            [["Timepoint", "Glucose"]]
            .sort_values("Timepoint")
            .reset_index(drop=True)
        )
        if x_df.empty:
            continue

        x_align = align_timeseries(x_df)
        x_align = apply_smoothing(x_align, lam=0.4)
        subj_entry["x"][em.lower()] = x_align

    # Derived: cgm_home_mean (home_cgm_1 & home_cgm_2) and cgm_all_mean (ctru_cgm + home_1 + home_2)
    home1 = subj_entry["x"].get("home_cgm_1", [])
    home2 = subj_entry["x"].get("home_cgm_2", [])
    ctru_cgm = subj_entry["x"].get("ctru_cgm", [])

    def mean_series(*series):
        non_empty = [s for s in series if s]
        if not non_empty:
            return []
        max_len = max(len(s) for s in non_empty)
        return [
            sum(s[i] for s in non_empty if i < len(s)) / sum(1 for s in non_empty if i < len(s))
            for i in range(max_len)
        ]

    subj_entry["x"]["cgm_home_mean"] = mean_series(home1, home2)
    subj_entry["x"]["cgm_all_mean"] = mean_series(ctru_cgm, home1, home2)

    validation_split[s] = subj_entry

with open(OUT / "validation_split.json", "w") as f:
    json.dump(validation_split, f, indent=4)
print(f"  Saved {len(validation_split)} subjects to {OUT/'validation_split.json'}")


# ---------- cgm_initial_cohort.csv for pretraining ----------
print("\nGenerating cgm_initial_cohort.csv (for SSL pretraining)...")

# Stanford initial cohort CGM
stanford_cgm = cgm[cgm["subject"].isin(initial_subjects)][["timestamp", "glucose_value", "subject"]].copy()
print(f"  Stanford initial cohort CGM rows: {len(stanford_cgm)} from {stanford_cgm['subject'].nunique()} subjects")

# Colas pretrain data — convert to same format (timestamp, glucose_value, subject)
colas_path = BASE / "Dataset" / "colas_pretrain.parquet"
if colas_path.exists():
    colas_df = pd.read_parquet(colas_path)
    colas_rows = []

    for _, row in colas_df.iterrows():
        sub_id = f"colas_{row['Subject_ID']}_{row['subsubject_id'].split('_')[-1]}"
        # Timestamps are time-of-day only; use a synthetic date per day
        glucose_list = np.array(row["Glucose_List"], dtype=float)
        # Interpolate NaN using linear interpolation
        if np.isnan(glucose_list).any():
            s = pd.Series(glucose_list).interpolate(method="linear", limit_direction="both")
            glucose_list = s.to_numpy()
        ts_list = row["Timestamp_List"]
        # Synthesize a date for each day window
        day_idx = int(row["subsubject_id"].split("_")[-1])
        base_date = pd.Timestamp(f"2000-01-{day_idx:02d}")
        timestamps = [base_date + pd.Timedelta(t) for t in ts_list]

        for ts, g in zip(timestamps, glucose_list):
            colas_rows.append({
                "timestamp": ts,
                "glucose_value": float(g),
                "subject": sub_id,
            })

    colas_cgm = pd.DataFrame(colas_rows)
    print(f"  Colas CGM rows: {len(colas_cgm)} from {colas_cgm['subject'].nunique()} subject-days ({colas_df['Subject_ID'].nunique()} unique subjects)")

    combined = pd.concat([stanford_cgm, colas_cgm], ignore_index=True)
else:
    print(f"  Warning: {colas_path} not found, skipping Colas data")
    combined = stanford_cgm

combined.to_csv(OUT / "cgm_initial_cohort.csv", index=False)
print(f"  Saved {len(combined)} total rows, {combined['subject'].nunique()} subjects -> {OUT/'cgm_initial_cohort.csv'}")

print("\nDone.")
