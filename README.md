# Baseline Predictive Pipeline -- ETAI

*Daniely da Silva Machado - 20260617*

This is the **starting point** for my semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: `python main.py [config]` trains + saves the model
├── predict.py             # `python predict.py [config]`: predict unseen data (config-driven)
├── config.yaml            # COMPAS config (default)
├── config_adult.yaml      # Adult-income config (week 9, Option A)
├── requirements.txt
├── src/
│   ├── data.py            # loading + cleaning (stateless rules)
│   ├── features.py        # feature engineering (fixed formulas)
│   ├── preprocessing.py   # imputation/encoding/scaling + feature/target & dev/test splits
│   ├── model.py           # model registry + feature selection + Pipeline assembly
│   ├── evaluate.py        # metrics from probabilities + fairness check
│   ├── tuning.py          # Optuna hyperparameter search + nested CV
│   └── results.py         # saves each run's report + the scoreboard
├── weeks-tasks/           # one file per week: what changed and why
├── models/                # trained pipelines (git-ignored)
├── results/               # generated per run: run_<config>_<date>.txt (git-ignored)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── adult_train.csv / adult_new.csv / ADULT_README.md
    └── README.md          # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Data diagnosis (EDA) | `clean_dataset()` driven by the diagnosis written as rules in `config.yaml` (`validity_rules`, `canonical_categories`, `placeholder_tokens`, `redundant_columns`): category cleanup, impossible values / placeholders → `NaN`, redundant-column removal |
| 4 | Leak-safe preprocessing + honest evaluation | Row-preserving cleaning + training-only `drop_duplicate_rows()`; `split_dev_test()` (locked test set) replacing `split_train_test`; `build_preprocessor()` (mechanism-matched imputation, target encoding, robust scaling — all fitted inside the `Pipeline`); stratified 5-fold CV (`cross_validate_pipeline`, out-of-fold classification report + fairness check); `dummy` and `random_forest` models |
| 7 | Boosting + evaluation by probabilities | `gradient_boosting`, `hist_gradient_boosting` and `naive_bayes` models; eight metrics scored from out-of-fold **probabilities** (ROC-AUC is the selection metric, chosen before comparing models); a **scoreboard** (`results/runs.csv`) keeps every run with the no-model floor and the untuned score; the corrected paired t-test separates real differences from noise |
| 8 | Feature selection + XGBoost | `xgboost` model (gradient boosting with per-split brakes: `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`); a `VoteSelector` feature-selection step (`min_votes` of a random forest, gradient boosting and an L1 logistic regression), switchable in `config.yaml` and tuned with the model (`select__min_votes`); a per-run cache folder (`tempfile.mkdtemp`) so tuning doesn't refit prep/select every trial; the kept features are printed and recorded on the scoreboard |
| 9 | Generalisation beyond COMPAS | `config_adult.yaml` (Option A: the same pipeline on a new dataset, Adult income); `main.py` now takes the config as an argument (`python main.py config_adult.yaml`), so there is a single entry point; a config-driven `prediction` step for `adult_new.csv`; `fairness_report` made dataset-agnostic (the COMPAS `score_text` comparison is now conditional); a regularised XGBoost default that clears all three target scores |
| 10 | Missing values + deployment | `_NUMERIC_IMPUTERS` (median / mean / knn / iterative / random_forest) with **scale-then-impute** in `build_preprocessor`; the chosen imputer is recorded on the scoreboard and in the report header; `predict.py` + a config-driven `prediction` section (`model_path`, `new_data_path`, `output_path`) save the final pipeline and predict unseen people |

## Model evaluation

The pipeline no longer trusts a single train/test split. It carves a **locked test set** aside once (20%, stratified, seed 42 — never used to fit or choose), and judges every candidate with **stratified 5-fold cross-validation** on the development set, fitting the *whole* pipeline (preprocessing + model) inside each fold so no fold leaks into its own preprocessing.

| Model | Holdout accuracy (1 split) | Holdout range, 30 seeds | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|---|
| Dummy (majority) | 0.550 | 0.550 – 0.550 | 0.549 ± 0.000 | 0.000 |
| Logistic regression | 0.672 | 0.658 – 0.697 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.602 | 0.558 – 0.624 | 0.611 ± 0.015 | +0.084 |
| Random forest | 0.655 | 0.623 – 0.674 | 0.652 ± 0.017 | +0.080 |

**Which number to trust: CV.** A single holdout is one draw from a distribution — for logistic regression the same model moves from 0.658 to 0.697 (~4 points) just by changing the seed, so a lone 0.672 is partly luck. CV uses every development row for validation and reports the mean and its spread (std), so it is both more stable and more honest. The week 2/3 "best model" conclusion still holds under CV: logistic regression remains the best model (highest CV accuracy) and the least overfit (gap ≈ 0 vs ~0.08 for the tree and forest).

Changing the scaler (`none` / `standard` / `minmax` / `robust`) leaves CV at ~0.672 (±0.013) — the difference is far smaller than the CV noise, so `robust` stays (a safe default for the skewed counts). KNN imputation was also tried and was slightly *worse* (0.669 vs 0.672) — the `_was_missing` flags already carry the missingness signal, so the median fill stays.

## Scoreboard (Week 7)

From week 7 the pipeline reports **eight metrics** computed from out-of-fold **probabilities**
(not a single 0/1 prediction), and optimises one **selection metric**: **F1**. F1 (the harmonic
mean of precision and recall) is the choice because recidivism has a clear positive class to
catch, both false positives and false negatives matter, and accuracy is a poor guide on this
imbalanced target — so a single balanced number that rewards detecting reoffenders (recall)
without flooding the flagged set (precision) beats threshold-free ranking.

Every `python main.py` run appends one row to `results/runs.csv` with the model, its settings,
the selection metric's mean/std and per-fold scores, plus the no-model floor (a dummy) and the
untuned score. Nested-CV (tuned) results, best first:

| Model | Tuned | F1 (mean ± std) | Untuned | Floor |
|---|---|---|---|---|
| HistGradientBoosting | ✓ | 0.644 ± 0.018 | 0.630 | 0.000 |
| Random forest | ✓ | 0.639 ± 0.018 | 0.606 | 0.000 |
| Logistic regression | ✓ | 0.633 ± 0.012 | 0.634 | 0.000 |
| Decision tree | ✓ | 0.633 ± 0.019 | 0.567 | 0.000 |
| Gradient boosting | ✓ | 0.620 ± 0.021 | 0.607 | 0.000 |
| AdaBoost | ✓ | 0.619 ± 0.016 | 0.599 | 0.000 |
| Oblique random forest | ✓ | 0.616 ± 0.011 | 0.596 | 0.000 |
| Naive Bayes | ✓ | 0.404 ± 0.013 | 0.403 | 0.000 |
| Dummy (floor) | ✗ | 0.000 ± 0.000 | 0.000 | 0.000 |

The corrected paired t-test on the same folds says HistGradientBoosting is not significantly
better than random forest, logistic regression or the decision tree, but it is better than
gradient boosting (p ≈ 0.04), AdaBoost and the oblique forest (p ≈ 0.02), and far better than
Naive Bayes (p ≈ 0.0001).

## Feature selection (Week 8)

Week 8 added **XGBoost** and a `VoteSelector` feature-selection step (a random forest,
gradient boosting and an L1 logistic regression vote; features with at least `min_votes` are
kept), switchable in `config.yaml`. To answer "does selection pay off?", XGBoost was tuned
(nested CV, 30 trials) **with** and **without** selection:

| Config | F1 (mean ± std) | Untuned | Features kept |
|---|---|---|---|
| With selection | 0.622 ± 0.015 | 0.590 | 2 (`age`, `priors_count`) |
| Without selection | 0.624 ± 0.012 | 0.604 | 10 (all) |

The corrected paired t-test says the two are indistinguishable (mean diff −0.002, p = 0.80).
On COMPAS — a clean dataset with ~10 mostly-informative features — selection buys nothing: the
2-feature model ties the full one, but is faster and far easier to explain. I'd keep selection
**off** for this dataset (it only adds training cost), and turn it on where features are noisy
or numerous, which is exactly where the notebook showed it paying off.

## Beyond COMPAS (Week 9)

Week 9 asked whether the pipeline only works for COMPAS, or actually generalises. **Option A** swaps in a
new dataset -- **Adult income** (1994 US Census / UCI): predict whether a person earns over $50K a year,
still a yes/no question. Run it with:

```bash
python main.py config_adult.yaml
```

This runs the pipeline on `config_adult.yaml` and, after refitting, predicts the 2,000 unseen people in
`data/adult_new.csv` (id + probability + 0/1 prediction → `results/adult_new_predictions.csv`).
`python main.py` (COMPAS) is unchanged.

**What broke, and how it was fixed**

| Broke | Why | Fix |
|---|---|---|
| `fairness_report` → `KeyError: 'score_text'` | Adult has no COMPAS `score_text` column | the COMPAS comparison now runs only when `score_text` is present |
| default XGBoost overfit (gap +0.12, accuracy 0.848) | small dev set (4,800 rows) | regularised tree: `max_depth 3`, `n_estimators 300`, `learning_rate 0.1` |
| tuning made F1 *worse* (0.669 → 0.655) | 15 trials overfit the inner folds | tuning left off; the regularised default already meets the targets |

**Results** (5-fold CV on the development set, three targets in the same run)

| Metric | Target | Floor | Reached |
|---|---|---|---|
| ROC-AUC | ≥ 0.90 | 0.50 | **0.911 ± 0.016** ✓ |
| Accuracy | ≥ 0.855 | 0.76 | **0.862 ± 0.010** ✓ |
| F1 (class 1, threshold 0.5) | ≥ 0.66 | 0.00 | **0.685 ± 0.030** ✓ |

Feature selection keeps 8 of 36 features (`age`, `education_num`, `capital_gain`, `capital_loss`,
`hours_per_week`, `marital_status`, `occupation`, `relationship`) -- every one of the 25 anonymised
`v`-columns, plus `workclass`, `native_country` and `race`, is voted out.

The hardest assumption to find was the fairness report's quiet COMPAS dependency: it read `score_text`  
commi  
clear all three. Feature selection paid off here (F1 0.685 vs 0.646 without it), the opposite of COMPAS,  
which is exactly the situation the week-8 notebook said selection is for: many noisy features.

## Missing values & deployment (Week 10)

Week 10 added model-based numeric imputation and made the pipeline deployable end-to-end.

**Imputation** (`src/preprocessing.py` → `_NUMERIC_IMPUTERS`). The numeric imputer is now a config
choice, and the numeric pipeline **scales first, then imputes** — so distance-based imputers compare
columns on one scale:

| `numeric_strategy` | What it does |
|---|---|
| `median` / `mean` | `SimpleImputer` (the defaults) |
| `knn` | `KNNImputer` — fills a gap from its `k` nearest neighbours |
| `iterative` | MICE with one completed dataset (`IterativeImputer`) |
| `random_forest` | missForest — `IterativeImputer` with a random-forest estimator |

Set it in `config.yaml` → `preprocessing.imputation.numeric_strategy`; the extra knobs are
`n_neighbors` (knn), and `max_iter` / `n_estimators` (iterative / random_forest). The chosen imputer
is printed, recorded on the scoreboard and in each run's report header.

**Deployment** (`predict.py` + a `prediction` section in `config.yaml`). After `main.py` refits the
final model, it saves it to `prediction.model_path`; `predict.py` loads it and predicts the rows in
`prediction.new_data_path` (same cleaning, no de-duplication, rows out = rows in):

```bash
python main.py                # train, evaluate, save models/model.joblib
python predict.py config.yaml # predict data/compas_new_people.csv -> results/compas_new_predictions.csv
```

The same flow works for Adult (`python predict.py config_adult.yaml`).

## Environment setup

You only need to do this once per machine.

### macOS / Linux

```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell

```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

If PowerShell blocks the activation script, run this once first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe

Same three steps as above, just with cmd's own activation command:

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**

```bash
source venv/bin/activate
python main.py
```

**Windows**

```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:

```bash
python main.py                    # COMPAS (config.yaml, the default)
python main.py config_adult.yaml  # Adult income (week 9)
```

The config file drives the whole run -- dataset, cleaning, preprocessing, model, tuning and
output are all read from it, so changing datasets or models is a config change, not a code
change. The pipeline loads and preprocesses the data, trains the model, and prints:

- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
