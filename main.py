"""
Entry point for the predictive pipeline.

Run with:
    python main.py                     # config.yaml (COMPAS by default)
    python main.py config_adult.yaml   # any config; everything else is read from it

The whole pipeline is config-driven: one config file chooses the dataset, cleaning rules,
preprocessing, model, tuning and output. It orchestrates:
    load config -> load data -> diagnose/clean -> drop duplicate rows (training only)
    -> split features/target -> lock the test set
    -> preprocessing + model as ONE Pipeline (src/model.py)
    -> stratified k-fold cross-validation of the whole pipeline, scored on several metrics
       from out-of-fold probabilities (week 7)
    -> hyperparameter tuning with Optuna, leak-free, on the development set only
    -> out-of-fold classification report + fairness check
    -> score the no-model floor and append the run to the scoreboard
    -> refit the final (tuned) model on the whole development set -> save results
    -> (optional) predict unseen data, when the config has a `prediction` section (week 9)
"""
import argparse
import os
import shutil
import tempfile
from datetime import datetime

import joblib
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from predict import predict
from src.data import clean_dataset, drop_duplicate_rows, load_data
from src.features import add_engineered_features
from src.preprocessing import split_dev_test, split_features_target
from src.model import build_pipeline
from src.evaluate import (
    build_scorers, cross_validate_pipeline, cv_report, oof_classification_report,
    fairness_report,
)
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "config.yaml"):
    """Runs the full pipeline for `config_path`, returning (config, run_id, final_pipeline).

    `final_pipeline` is the model refit on the whole development set. If the config has a
    `prediction` section, the unseen rows are predicted and saved at the end (week 9).
    """
    config = load_config(config_path)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    model_type = config["model"]["type"]

    # load + diagnose-and-clean: nothing here is learned from the data, so it's safe to run
    # on the whole dataset -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    # training data only: the same person must not count twice, or sit in both dev and test
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))
    # fixed-formula features (week 6 challenge): nothing learned, so safe before the split
    df_clean = add_engineered_features(df_clean, config["preprocessing"].get("engineered_features"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # the final test set is set aside HERE and never used again in this script
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away (not used)")

    # preprocessing + (optional) feature selection + model = one estimator: anything that learns
    # is re-fit in every fold. The cache folder lets tuning reuse the fitted prep/select steps
    # across trials that only change the model's settings (week 8).
    selection_config = config.get("feature_selection")
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")
    pipeline = build_pipeline(config["preprocessing"], config["model"], selection_config,
                              memory=cache_dir)

    # the no-model floor: a dummy scored on the same folds, for every run to be measured against
    # (no selection here: the floor is just a baseline, and selection before a dummy buys nothing)
    floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]},
                                              selection_metric, n_jobs)
    floor = floor_scores["validation"].mean()

    # default-hyperparameter CV of the whole pipeline
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers,
                                                     selection_metric, n_jobs)
    untuned = fold_scores["validation"].mean()

    print(f"\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    # hyperparameter tuning -- development set only (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        # feature selection's `select__min_votes` is tuned together with the model (week 8)
        if selection_config and selection_config.get("enabled", False):
            search_space.update(selection_config.get("search_space", {}))
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 1. the honest estimate of the procedure (its out-of-fold predictions feed the reports)
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            if isinstance(value, float):
                print(f"    {name.replace('model__', ''):<20s} {value:.4g}")
            else:
                print(f"    {name.replace('model__', ''):<20s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    # the model we'd actually use: same pipeline, refit on EVERY development row
    pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    report += "\n" + refit + "\n"

    # which features the final model keeps (week 8)
    if selection_config and selection_config.get("enabled", False):
        kept = pipeline.named_steps["select"].get_feature_names_out(
            input_features=pipeline.named_steps["prep"].get_feature_names_out())
        print(f"Features kept ({len(kept)}): {', '.join(kept)}")
        report += f"Features kept ({len(kept)}): {', '.join(kept)}\n"

    # one row per run on the scoreboard, with the no-model floor and the untuned score
    fs_label = ("on (min_votes=" + str(selection_config.get("min_votes", 2)) + ")"
                if selection_config and selection_config.get("enabled", False) else "off")
    row = {"run": run, "data": config["data"]["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
           "imputer": config["preprocessing"].get("imputation", {}).get("numeric_strategy", "median"),
           "feature_selection": fs_label,
           "params": str(config["model"].get("params")),
           "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "encoder", "scaler", "imputer", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report,
                    config_name=config_name)
    shutil.rmtree(cache_dir, ignore_errors=True)   # week 8: delete this run's cache folder
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")

    # optional prediction step (week 9): save the trained model and predict unseen data,
    # when the config declares a `prediction` section
    if config.get("prediction"):
        model_path = config["prediction"]["model_path"]
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f"Model saved -> {model_path}")
        predict(config_path)

    return config, run, pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the predictive pipeline. Everything is read from one config file.")
    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="path to a config YAML (default: config.yaml)")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
