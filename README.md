# BCIS 566 Final Project

This repository contains the reproducible Python script for the BCIS 566 final project analysis of Glassdoor employee reviews. The script builds a recommendation target, evaluates predictive models, saves paper-ready figures, exports result tables, and includes a firm-level heterogeneity extension using both fixed-effects interaction and multilevel logistic models.

## Files

| File | Purpose |
|---|---|
| `BCIS_566_Final_Project.py` | Main reproducible Python script for submission |
| `README.md` | Project overview and run instructions |

**Dataset:** David Gauthier, Glassdoor Job Reviews, Kaggle:
https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews

## Setup

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn factor_analyzer vaderSentiment adjustText
```

The script expects the Glassdoor review dataset to contain the columns used in the analysis, including recommendation status, structured rating variables, review text fields, firm names, reviewer status, and review dates.

## Run locally

```bash
python BCIS_566_Final_Project.py \
  --data-path /path/to/glassdoor_reviews.csv \
  --output-dir figures \
  --table-dir tables
```

## Run in Google Colab after mounting Drive manually

```bash
python BCIS_566_Final_Project.py \
  --data-path "/content/drive/MyDrive/glassdoor_reviews.csv" \
  --output-dir "/content/drive/MyDrive/project_figures" \
  --table-dir "/content/drive/MyDrive/project_tables" \
  --no-mount-drive
```

Use `--no-mount-drive` when Drive has already been mounted manually in Colab. The script also attempts to handle Drive-mount errors safely and continue.

## Main command-line options

| Option | Default | Purpose |
|---|---:|---|
| `--data-path` | `/content/drive/MyDrive/glassdoor_reviews.csv` | Location of the input CSV file |
| `--output-dir` | `figures` | Folder where PNG figures are saved |
| `--table-dir` | `tables` | Folder where CSV result tables are saved |
| `--cv-max-rows` | `75000` | Stratified sample size used for cross-validation; use `0` for full training data |
| `--learning-curve-max-rows` | `75000` | Stratified sample size used for learning curves; use `0` for full training data |
| `--no-mount-drive` | off | Skip automatic Google Drive mounting |

## Output folders

The script creates the figure and table output folders automatically. Figures are saved as 300-DPI PNG files. Result tables are saved as CSV files.

The generated outputs are documented in:

- `FIGURE_MANIFEST.md`
- `figure_manifest.csv`
- `TABLE_MANIFEST.md`
- `table_manifest.csv`

## Analysis overview

The script performs the following steps:

1. Loads and filters Glassdoor reviews to clear recommendation outcomes.
2. Constructs the binary dependent variable where positive recommendation equals 1.
3. Creates reviewer-status and review-year controls.
4. Runs exploratory data analysis and saves the class-balance figure.
5. Splits the data into stratified train/test sets.
6. Runs cross-validation for the dummy classifier, logistic regression, and random forest models.
7. Evaluates the main models on the test set.
8. Saves model interpretation plots for logistic coefficients and random-forest feature importance.
9. Examines shared variance among structured workplace rating variables.
10. Estimates factor-analysis and sentiment-augmented logistic models.
11. Runs TF-IDF text models for pros and cons text.
12. Estimates firm-level heterogeneity models.
13. Saves appendix diagnostics including learning curves, calibration, precision-recall, and additional model-specific ROC/confusion-matrix plots.

## Firm-level heterogeneity extension

To examine whether negative employee voice was more consequential for some firms than for others, the script estimates two additional models using cons sentiment as the focal negative-signal variable.

First, it fits a firm fixed-effects interaction logistic model in which the slope of standardized cons sentiment is allowed to vary by firm. Second, it fits a multilevel logistic model with random intercepts and random slopes for standardized cons sentiment by firm. In both models, recommendation remains the dependent variable, while the latent workplace-evaluation factor, reviewer status, and review year are retained as controls.

The multilevel model is fit using variational Bayes. If model warnings occur during estimation, the script prints the warning messages and notes that exact posterior estimates should be interpreted cautiously.

## Reproducibility notes

The script uses `RANDOM_STATE = 42` for train/test splitting, model fitting where applicable, cross-validation sampling, learning-curve sampling, and firm-level subsampling. The default cross-validation and learning-curve row caps are intended to keep runtime manageable on the full Glassdoor dataset.
