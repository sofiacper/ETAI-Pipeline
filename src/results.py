"""
Saving each run's results to disk.

Writes each run's full report (cross-validation scores, classification report, fairness
table) to a timestamped file in `results/`, and appends one row per run to the scoreboard
CSV (`results/runs.csv`), so runs can be compared after changing config.yaml.
"""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str,
             config_name: str = "config") -> str:
    """
    Writes one run's full report to `results_dir/run_<config_name>_<run>.txt` (the folder
    is created automatically if it doesn't exist yet) and returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{config_name}_{run}.txt")

    cv = config["cv"]
    header = (
        f"Config: {config_name}\n"
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}  "
        f"imputer={config['preprocessing'].get('imputation', {}).get('numeric_strategy', 'median')}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}  selection_metric={cv.get('selection_metric', 'accuracy')}  "
        f"metrics={cv.get('metrics')}\n"
    )
    fs = config.get("feature_selection", {})
    header += ("Feature selection: "
               + (f"enabled (min_votes={fs.get('min_votes', 2)})" if fs.get("enabled", False) else "disabled")
               + "\n")
    tuning = config.get("tuning", {})
    if tuning.get("enabled", False):
        header += (f"Tuning: {tuning.get('n_trials')} Optuna trials  "
                   f"n_splits={tuning.get('n_splits')}  random_state={tuning.get('random_state')}\n")
    header += (
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
