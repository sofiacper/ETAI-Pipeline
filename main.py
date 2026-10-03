"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> split features/target
    -> leak-safe train/test split -> preprocess + train (week 3's encoder/scaler pair)
    -> evaluate (accuracy, fairness) -> save results
"""
import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
    build_preprocessor
)
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run

def main():
    # 1. Read configurations
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # 2. Load data
    print("Loading data...")
    df_raw = load_data(config["data"]["path"])

    # 3. Clean data (keeping all rows)
    print("Cleaning data...")
    df_clean = clean_dataset(df_raw, config.get("diagnostics", {}))

    # 4. Remove duplicate rows (training data only)
    print("Removing duplicates...")
    df_clean = drop_duplicate_rows(df_clean, config.get("diagnostics", {}).get("id_column"))

    # 5. Split features (X) and target (y)
    print("Splitting features and target...")
    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        config.get("preprocessing", {}).get("mnar_indicator_sources", [])
    )

    # 6. Dev/Test Split (Holdout Method)
    print("Splitting into Dev and Test sets...")
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"]
    )

    # 7. Build Pipeline (Preprocessing + Model)
    print("Building pipeline...")
    pipe = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"]))
    ])

    # 8. Train model
    print("Training model...")
    pipe.fit(X_train, y_train)

    # 9. Evaluate
    print("\n--- EVALUATION ---")
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    report_text = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    fairness_text = fairness_report(y_test, y_test_pred, extras_test, config["data"]["sensitive_attr"])

    # 10. Save results
    full_report = report_text + "\n\n" + fairness_text
    save_path = save_run(config["output"]["results_dir"], config, full_report)
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()