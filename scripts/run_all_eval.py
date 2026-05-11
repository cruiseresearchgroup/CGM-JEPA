"""
Run downstream evaluation across the 3 paper regimes and both endpoints.

Settings (correspond to paper Tables 1-6):
  1. initial_to_validation, ctru_venous -> ctru_venous       (cohort generalization, venous)  → Tables 5, 6
  2. validation_to_validation, ctru_venous -> cgm_home_mean  (venous → home CGM transfer)     → Tables 3, 4
  3. validation_to_validation, cgm_home_mean -> cgm_home_mean (home CGM in-domain)            → Tables 1, 2

Metabolic outcomes: ir, beta
Total: 3 settings × 2 outcomes = 6 runs

Usage:
    python scripts/run_all_eval.py
    python scripts/run_all_eval.py --dry-run
"""
import subprocess
import sys
import json
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
log_file = f"logs/eval_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

SETTINGS = [
    {
        "label": "in_domain_venous",
        "pipeline": "initial_to_validation",
        "extract_method": "ctru_venous",
        "val_extract_method": "ctru_venous",
    },
    {
        "label": "venous_to_home_cgm",
        "pipeline": "validation_to_validation",
        "extract_method": "ctru_venous",
        "val_extract_method": "cgm_home_mean",
    },
    {
        "label": "home_cgm_in_domain",
        "pipeline": "validation_to_validation",
        "extract_method": "cgm_home_mean",
        "val_extract_method": "cgm_home_mean",
    },
]

METABOLIC = ["ir", "beta"]


def write_overrides(overrides, path=None):
    if path is None:
        # Anchor to project root so this works regardless of where the user
        # invoked the script from.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "config", "_eval_overrides.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2)
    return path


def run_eval(overrides, label):
    override_path = write_overrides(overrides)
    log.info(f"{'='*70}")
    log.info(f"  {label}")
    log.info(f"  pipeline={overrides['pipeline']}, metabolic={overrides['metabolic']}")
    log.info(f"  extract={overrides['extract_method']} -> val_extract={overrides['val_extract_method']}")
    log.info(f"{'='*70}")

    # Subprocess must run from the project root so `python -m eval.class_reg`
    # can find the `eval` package (this script lives under scripts/).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-m", "eval.class_reg", "--ablation_overrides", override_path],
        cwd=project_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    for line in result.stdout.splitlines():
        log.info(f"  {line}")

    if result.returncode != 0:
        log.error(f"  FAILED with exit code {result.returncode}")

    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Print encoder versions that will be used across all eval runs.
    from config.config_downstream import config as _dc
    log.info("=" * 70)
    log.info("  ENCODER VERSIONS (from config_downstream.py)")
    log.info("=" * 70)
    for key in ["cgm_jepa_version", "cgm_jepa_glu_version", "gluformer_version",
                "cgm_ts2vec_version", "cgm_mantis_version", "cgm_moment_version"]:
        log.info(f"  {key:22s} = {_dc.get(key, '<not set>')}")
    log.info("=" * 70)

    # Need --ablation_overrides support in class_reg.py
    results = []

    for setting in SETTINGS:
        for metabolic in METABOLIC:
            label = f"{setting['label']}_{metabolic}"
            overrides = {
                "pipeline": setting["pipeline"],
                "extract_method": setting["extract_method"],
                "val_extract_method": setting["val_extract_method"],
                "metabolic": metabolic,
                "wandb_run_name": label,
            }

            if args.dry_run:
                log.info(f"[DRY RUN] {label}: {json.dumps(overrides)}")
                continue

            success = run_eval(overrides, label)
            results.append({
                "label": label,
                "status": "ok" if success else "FAILED",
                **overrides,
            })

    if not args.dry_run:
        log.info(f"\n{'='*70}")
        log.info("  EVALUATION SUMMARY")
        log.info(f"{'='*70}")
        for r in results:
            status = "ok" if r["status"] == "ok" else "FAIL"
            log.info(f"  [{status}] {r['label']}")

        with open("logs/eval_all_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Summary saved to logs/eval_all_summary.json")
        log.info(f"Full log saved to {log_file}")


if __name__ == "__main__":
    main()
