# Baseline Predictive Pipeline -- ETAI

20231723 Tiago Mota

## CONCLUSIONS 

# Week 2
We can see that in the logistic regression we have a train accuracy score of 0.677 and a training of 0.675. On the other hand, using decision trees, we get training accuracy score of 0.829 and 0.626 in the test. We can notice a much bigger difference of scores using the decision tree. This means that it memorized the data rather than doing what linear regression will do better, that is generalize better for unseen data. This is overfitting. 

We can also see that the results for precision, recall and F1 score were respectively: 
0.61, 0.48, 0.54 - decision trees
0.66, 0.6, 0.63 - logistic regression

Here we can also notice that when using the logistic regression model we get both precision and recall higher values, which also leads to a higher F1 score and which means that this will lead to a higher rate of correct results.

----------------------
# Week 3
We can see that in last week, in the decision tree, the tree was completely memorizing the training data leading to massive overfitting. The train accuracy was 82.9% but the test accuracy was only 62.6% (a giant gap of +0.2). 
We see that we get different results from when we add the max depth = 5 and when there is no max depth defined. Train Accuracy	0.792 (79.2%) for the unrestricted tree and	0.684 (68.4%) for the restricted one. Test Accuracy	of 0.611 (61.1%) in the unrestricted and	0.665 (66.5%) for the restricted. This means we get a smaller discrepancy in the restricted on of +0.02, which is way better. This is because by adding max_depth: 5, we stopped it from growing too deep.

In the logistic regreesion the test accuracy went down slightly (from 67.5% to 65.7%). It just means the model is now being tested on a more realistic dataset instead of a perfectly filtered one.

In the previous week, the race categories were a mess. We had duplicates like "African-American" vs "AFRICAN-AMERICAN" and extra signs like "-" or "?". Now, the pipeline cleans all of this up, groups everything correctly into 6 categories, and turns the weird symbols into missing values.

Because of this, now we can make some interpretations: The original COMPAS score has a False Positive Rate (FPR) of 44% for African-Americans (meaning 44% of innocent people in this group were predicted to reoffend), compared to only 24% for Caucasians. We managed to drop the FPR for African-Americans down to 28%. There is still a gap (Caucasian FPR is 14-16%), which means the historical bias in the dataset is still there, but our model is already way less biased than the actual system used in courts.

---------------
# Week 4

(This is using holdout method only (80-20), cross validadtion is only used next week)

When using logistic regression we got training accuracy of 67.6% and a test accuracy of 65.8%. The gap between the two was only +0.018 and so there is no overfitting. When talking about random forest, we clearly got overfitted results.It reached a 72.8% training accuracy, but test performance dropped to 64.4% (lower than the Logistic Regression). The gap spiked to +0.084.

(Precision vs. Recall)

  - Logistic Regression:When predicting an individual would reoffend (class 1), it was correct 66% of the time (precision = 0.66), though it missed more than half of the actual reoffenders (recall = 0.48).

  - Random Forest: Class 1 recall up to 0.59.precision dropped to 0.61, meaning the algorithm triggered false alarms much more frequently to capture that larger volume.


Fairness Analysis and Demographic Impact (FPR)

  - Logistic Regression: Significantly reduced demographic disparities. The False Positive Rate (share of people who did not reoffend but were incorrectly predicted to do so) dropped to 28% for the African-American population and 14% for the Caucasian population.

  - Random Forest: Dangerously mirrored the original bias of the COMPAS system. To achieve its higher recall, the model sacrificed true negatives, driving the False Positive Rate up to 42% for the African-American population (nearly matching COMPAS's own 44%) and 23% for the Caucasian population.

  
  For now, we are still chosing to use LOGISTIC REGRESSION.

----------------------
# Week 5 & 6
(We upgraded from a simple holdout method to a normal cross vaidation and then robust Nested Cross-Validation. We also implemented automated hyperparameter tuning using Optuna to chose the best hyperparameters and optimize our models.)

When running the Random Forest with default parameters, we still see the same overfitting problem from Week 4: training accuracy hit 73.3%, but validation dropped to 65.0% (a gap of +0.082).
However, after using Optuna to tune the hyperparameters (like limiting tree depth and requiring more samples per leaf), the overfitting was decreased by a lot. The training accuracy was at 69.0% and the validation accuracy increased to 67.9%. The gap shrank to just +0.011, giving us a highly stable model.

(Precision vs. Recall)

Tuned Random Forest: When predicting an individual would reoffend (class 1), the model is now correct 67% of the time (precision = 0.67), which is better than last week's Logistic Regression. It successfully captures 56% of the actual reoffenders (recall = 0.56). It strikes a much healthier balance, avoiding the massive false alarms we saw in the untuned forest.
git 
Fairness Analysis and Demographic Impact (FPR)

The original commercial system heavily penalizes the African-American population with a 45% False Positive Rate (predicting they will reoffend when they actually don't), compared to just 23% for the Caucasian population.

Tuned Random Forest: Tuning the hyperparameters drastically improved fairness compared to our Week 4 attempt. The False Positive Rate dropped from a dangerous 42% down to 29% for the African-American population, and down to 16% for the Caucasian population. While the historical bias is still visible, our tuned model is now significantly fairer than the original COMPAS software.

For now, thanks to the hyperparameter tuning, we are upgrading and choosing to use the TUNED RANDOM FOREST.

-------------------------
# Week 7
Our tuned AdaBoost model gave us the best and most stable results so far. Our new main score best is 0.726. There is no sign of overfitting. The gap between training and testing is incredibly small (just +0.017). The overall accuracy is 68.2%, beating last week's Random Forest.

(Precision vs. Recall)

Tuned AdaBoost: When the model says someone will reoffend, it is right 68% of the time (precision). It successfully catches 56% of the real reoffenders (recall). This is a great balance because it catches many cases without making too many false alarms.

Fairness Analysis (False Positive Rate)

COMPAS Baseline: The original system was very unfair. It wrongly flagged 45% of innocent African-Americans as a risk, but only 23% of innocent Caucasians.

Tuned AdaBoost: Our new model keeps the great fairness results from last week. The error rate dropped to 29% for African-Americans and 17% for Caucasians. The historical bias is still there, but our model is much fairer than the real COMPAS software.

On our new automatic scoreboard, AdaBoost is in 1st place with a score of 0.726 (beating the basic, untuned version).

For now, we are upgrading and choosing to use the TUNED ADABOOST.

----------------------
# Week 8

This week, we added Feature Selection to remove useless data before training. Our tuned AdaBoost model with Feature Selection gave us very good results. Our new main score (ROC-AUC) is 0.723. There is no sign of overfitting. The gap between training and testing is incredibly small (just +0.012). The overall accuracy is 67.9%, remaining very close to last week's peak, but now using a simpler, lighter model.

(Precision vs. Recall)

Tuned AdaBoost (with Feature Selection): When the model says someone will reoffend, it is right 67% of the time (precision). It successfully catches 56% of the real reoffenders (recall). This is a great balance because it catches many cases without making too many false alarms, using even less data than before.

Fairness Analysis (False Positive Rate)

COMPAS Baseline: The original system was very unfair. It wrongly flagged 45% of innocent African-Americans as a risk, but only 23% of innocent Caucasians.

Tuned AdaBoost (with Feature Selection): Our new model keeps the great fairness results from last week. The error rate shifted slightly to 30% for African-Americans and 16% for Caucasians. The historical bias is still there, but our model is much fairer than the real COMPAS software.

On our automatic scoreboard, last week's AdaBoost (without selection) is technically still in 1st place with a score of 0.726, but our new model is right behind it at 0.723. Because it scores almost the same while using fewer features, it is actually a more efficient algorithm.

For now, we are upgrading and choosing to use the TUNED ADABOOST WITH FEATURE SELECTION.

---------------------
# WEEK 9

The tuned Hist Gradient Boosting model achieved good results, hitting all the target scores for this challenge. Our main score (ROC-AUC) reached 0.906. The model is highly stable, showing a very small train-validation gap of just +0.016. The overall accuracy is 85.6%, proving the model adapted perfectly to a completely new problem.

(Precision vs. Recall)

Tuned Hist Gradient Boosting: When the model predicts someone makes over $50K, it is correct 76% of the time (precision). It successfully catches 58% of the actual high earners (recall). This is a strong balance for a highly imbalanced dataset where only 24% of people earn over $50K.

Fairness Analysis (False Positive Rate)
Model Fairness by Sex: Since we are using a new dataset, we now evaluate fairness based on gender instead of race. The False Positive Rate (predicting someone earns >$50K when they actually don't) is very low overall. The error rate is just 8% for Males and 2% for Females. (Note: The original COMPAS baseline does not apply to this financial dataset).


-----------------------
(para dar push e pull faço: 
git add .
git commit -m "something"
git push)

--------------------

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

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
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
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
