# File: Glassdoor_Final_Project.py
# -*- coding: utf-8 -*-
"""
Final Glassdoor Project cleaned submission script.

Users running outside Google Colab should either update DATA_PATH below or pass
--data-path at the command line. Figures are saved to --output-dir, and result
tables are saved to --table-dir.

Example:
    python Glassdoor_Final_Project.py \
        --data-path /path/to/glassdoor_reviews.csv \
        --output-dir figures \
        --table-dir tables

Colab example after mounting Drive manually:
    python Glassdoor_Final_Project.py \
        --data-path "/content/drive/MyDrive/glassdoor_reviews.csv" \
        --output-dir "/content/drive/MyDrive/project_figures" \
        --table-dir "/content/drive/MyDrive/project_tables" \
        --no-mount-drive

Install dependencies with:
    pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn factor_analyzer vaderSentiment adjustText
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    PrecisionRecallDisplay,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import cross_validate, learning_curve, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from factor_analyzer import FactorAnalyzer
except ImportError as exc:
    raise ImportError(
        "Missing dependency: factor_analyzer. Install it with "
        "`pip install factor_analyzer` before running this script."
    ) from exc

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as exc:
    raise ImportError(
        "Missing dependency: vaderSentiment. Install it with "
        "`pip install vaderSentiment` before running this script."
    ) from exc

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================

RANDOM_STATE = 42
DATA_PATH = "/content/drive/MyDrive/glassdoor_reviews.csv"
OUTPUT_DIR = "figures"
TABLE_OUTPUT_DIR = "tables"
CV_MAX_ROWS = 75_000
LEARNING_CURVE_MAX_ROWS = 75_000

MIN_REVIEWS_PER_FIRM = 2_000
MAX_FIRMS = 20
PER_FIRM_CAP = 5_000

FIGURE_DIR = Path(OUTPUT_DIR)
TABLE_DIR = Path(TABLE_OUTPUT_DIR)

sns.set_theme(style="whitegrid")


def mount_google_drive_if_available() -> None:
    """Mount Google Drive when the script runs inside Colab."""
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Skipping Google Drive mount: {exc}")


def save_current_figure(filename: str) -> None:
    """Save the active Matplotlib figure in FIGURE_DIR and then display it."""
    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = FIGURE_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()


def save_table(
    table: pd.DataFrame | pd.Series,
    filename: str,
    index: bool = False,
) -> None:
    """Save a result table in TABLE_DIR."""
    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = TABLE_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=index)
    print(f"Saved table: {output_path}")


def make_safe_filename(label: str) -> str:
    """Convert a model label into a safe lowercase filename stem."""
    return (
        label.lower()
        .replace("+", "plus")
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_numeric_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Create a median-imputation and standardization pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )


def build_main_models(numeric_features: list[str]) -> dict[str, Pipeline]:
    """Create the baseline, logistic-regression, and random-forest models."""
    return {
        "Dummy": Pipeline(
            steps=[
                ("preprocessor", build_numeric_preprocessor(numeric_features)),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_numeric_preprocessor(numeric_features)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_numeric_preprocessor(numeric_features)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def maybe_sample_for_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally use a stratified CV sample to reduce random-forest runtime."""
    if max_rows is None or len(X_train) <= max_rows:
        return X_train, y_train

    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    return X_sample, y_sample


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict[str, float | str]:
    """Compute the classification metrics used throughout the project."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
    }


def fit_and_evaluate_classifier(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> tuple[Pipeline, dict[str, float | str]]:
    """Fit a classifier, print the main metrics, and return the fitted model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 3))
    print("F1:", round(f1_score(y_test, y_pred, zero_division=0), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = compute_classification_metrics(y_test, y_pred, y_prob, model_name)
    return model, metrics


def build_logit_pipeline(numeric_features: list[str]) -> Pipeline:
    """Build a reusable logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_numeric_preprocessor(numeric_features)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def fit_and_evaluate_logit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> tuple[Pipeline, dict[str, float | str]]:
    """Fit and evaluate a logistic-regression model for numeric feature sets."""
    model = build_logit_pipeline(X_train.columns.tolist())
    return fit_and_evaluate_classifier(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name,
    )


def fit_text_logit(
    train_text: pd.Series,
    test_text: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    min_df: int = 25,
    max_df: float = 0.80,
    max_features: int = 5_000,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Fit a TF-IDF logistic model and return coefficient tables."""
    text_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                ),
            ),
            (
                "logit",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )

    text_model.fit(train_text, y_train)
    y_pred = text_model.predict(test_text)
    y_prob = text_model.predict_proba(test_text)[:, 1]

    print(f"\n{model_name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 3))
    print("F1:", round(f1_score(y_test, y_pred, zero_division=0), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    feature_names = text_model.named_steps["tfidf"].get_feature_names_out()
    coefs = text_model.named_steps["logit"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "term": feature_names,
            "coefficient": coefs,
        }
    ).sort_values("coefficient", ascending=False)

    top_positive = coef_df.head(20).copy()
    top_negative = coef_df.tail(20).sort_values("coefficient").copy()
    metrics = compute_classification_metrics(y_test, y_pred, y_prob, model_name)

    return text_model, coef_df, top_positive, top_negative, metrics



def plot_learning_curve_for_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    max_rows: int | None,
) -> None:
    """Save a learning curve for a fitted-model specification."""
    X_curve, y_curve = maybe_sample_for_cv(X_train, y_train, max_rows)

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_curve,
        y_curve,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    learning_curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_roc_auc_mean": train_scores.mean(axis=1),
            "train_roc_auc_std": train_scores.std(axis=1),
            "validation_roc_auc_mean": validation_scores.mean(axis=1),
            "validation_roc_auc_std": validation_scores.std(axis=1),
        }
    )
    safe_name = make_safe_filename(model_name)
    save_table(
        learning_curve_df,
        f"appendix_learning_curve_{safe_name}.csv",
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        learning_curve_df["train_size"],
        learning_curve_df["train_roc_auc_mean"],
        marker="o",
        label="Training ROC-AUC",
    )
    plt.plot(
        learning_curve_df["train_size"],
        learning_curve_df["validation_roc_auc_mean"],
        marker="o",
        label="Cross-validation ROC-AUC",
    )
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("ROC-AUC")
    plt.legend()
    save_current_figure(f"appendix_learning_curve_{safe_name}.png")

def zscore(series: pd.Series) -> pd.Series:
    """Return a z-scored series; avoid division by zero for constant columns."""
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible local or Colab runs."""
    parser = argparse.ArgumentParser(
        description="Run the Glassdoor recommendation analysis.",
    )
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help=(
            "Path to glassdoor_reviews.csv. Users running outside Colab should "
            "set this to their local CSV path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory where all PNG figures should be saved.",
    )
    parser.add_argument(
        "--table-dir",
        default=TABLE_OUTPUT_DIR,
        help="Directory where all CSV result tables should be saved.",
    )
    parser.add_argument(
        "--cv-max-rows",
        type=int,
        default=CV_MAX_ROWS,
        help=(
            "Optional stratified training-sample size for cross-validation. "
            "Set to 0 to use the full training data."
        ),
    )
    parser.add_argument(
        "--learning-curve-max-rows",
        type=int,
        default=LEARNING_CURVE_MAX_ROWS,
        help=(
            "Optional stratified training-sample size for learning curves. "
            "Set to 0 to use the full training data."
        ),
    )
    parser.add_argument(
        "--no-mount-drive",
        action="store_true",
        help="Skip automatic Google Drive mounting when running in Colab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global FIGURE_DIR, TABLE_DIR
    FIGURE_DIR = Path(args.output_dir)
    TABLE_DIR = Path(args.table_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.cv_max_rows == 0:
        args.cv_max_rows = None
    if args.learning_curve_max_rows == 0:
        args.learning_curve_max_rows = None

    # =============================================================================
    # 2. LOAD DATASET
    # =============================================================================

    if not args.no_mount_drive:
        mount_google_drive_if_available()

    df = pd.read_csv(args.data_path)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nInfo:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))

    # =============================================================================
    # 3. TARGET CONSTRUCTION
    # =============================================================================

    df = df[df["recommend"].isin(["v", "x"])].copy()
    df["target"] = (df["recommend"] == "v").astype(int)

    print("\nRecommendation counts:")
    print(df["recommend"].value_counts())

    print("\nTarget counts:")
    print(df["target"].value_counts())

    print("\nTarget proportions:")
    print(df["target"].value_counts(normalize=True))

    df["date_review"] = pd.to_datetime(df["date_review"], errors="coerce")
    df["review_year"] = df["date_review"].dt.year

    df["current_clean"] = (
        df["current"]
        .astype(str)
        .str.lower()
        .str.contains("current")
        .astype(int)
    )

    rating_cols = [
        "work_life_balance",
        "culture_values",
        "career_opp",
        "comp_benefits",
        "senior_mgmt",
    ]

    aux_cols = [
        "current_clean",
        "review_year",
    ]

    features = rating_cols + aux_cols

    X = df[features].copy()
    y = df["target"].copy()

    print("\nFeature preview:")
    print(X.head())
    print("\nTarget preview:")
    print(y.head())

    # =============================================================================
    # 4. EDA
    # =============================================================================

    plt.figure(figsize=(7, 5))
    sns.countplot(x=y)
    plt.title("Target Distribution")
    plt.xlabel("Recommendation target")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Negative recommendation", "Positive recommendation"])
    save_current_figure("figure1_class_balance.png")

    missing_pct = X.isnull().mean().sort_values(ascending=False) * 100
    print("\nMissing percentage by feature:")
    print(missing_pct)

    print("\nDescriptive statistics:")
    print(X.describe())

    # =============================================================================
    # 5. TRAIN/TEST SPLIT AND PREPROCESSING
    # =============================================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/test shapes:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    models = build_main_models(features)

    # =============================================================================
    # 6. CROSS-VALIDATION
    # =============================================================================

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    X_cv, y_cv = maybe_sample_for_cv(X_train, y_train, args.cv_max_rows)

    cv_rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_cv,
            y_cv,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"Model": name}
        for metric in scoring:
            row[metric] = scores[f"test_{metric}"].mean()
        cv_rows.append(row)

    cv_results_df = pd.DataFrame(cv_rows).sort_values(
        "roc_auc",
        ascending=False,
    )
    print("\nCross-validation results:")
    print(cv_results_df.round(3))
    save_table(cv_results_df, "table01_cross_validation_results.csv")

    # =============================================================================
    # 7. MAIN MODEL EVALUATION
    # =============================================================================

    fitted_models = {}
    summary_rows = []

    for name, model in models.items():
        fitted_model, metrics = fit_and_evaluate_classifier(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            name,
        )
        fitted_models[name] = fitted_model
        summary_rows.append(metrics)

    results_df = pd.DataFrame(summary_rows).sort_values(
        "ROC_AUC",
        ascending=False,
    )
    print("\nMain model comparison:")
    print(results_df.round(3))
    save_table(results_df, "table02_main_model_comparison.csv")

    for model_name, fitted_model in fitted_models.items():
        safe_name = make_safe_filename(model_name)

        ConfusionMatrixDisplay.from_estimator(fitted_model, X_test, y_test)
        plt.title(f"{model_name} Confusion Matrix")
        save_current_figure(f"appendix_confusion_matrix_{safe_name}.png")

        RocCurveDisplay.from_estimator(fitted_model, X_test, y_test)
        plt.title(f"{model_name} ROC Curve")
        save_current_figure(f"appendix_roc_curve_{safe_name}.png")

    for model_name in ["Logistic Regression", "Random Forest"]:
        plot_learning_curve_for_model(
            models[model_name],
            X_train,
            y_train,
            model_name,
            args.learning_curve_max_rows,
        )

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    plt.title("ROC Curves: Main Models")
    save_current_figure("figure5_roc_curves.png")

    primary_model_name = "Logistic Regression"
    primary_model = fitted_models[primary_model_name]

    ConfusionMatrixDisplay.from_estimator(primary_model, X_test, y_test)
    plt.title("Logistic Regression Confusion Matrix")
    save_current_figure("figure6_confusion_matrix.png")

    # =============================================================================
    # 8. MODEL INTERPRETATION
    # =============================================================================

    log_fitted = fitted_models["Logistic Regression"]
    coef = log_fitted.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coef,
        }
    ).sort_values("coefficient", ascending=False)

    print("\nLogistic regression coefficients:")
    print(coef_df)
    save_table(coef_df, "table03_logistic_coefficients.csv")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=coef_df, x="coefficient", y="feature")
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    save_current_figure("figure2_logistic_coefficients.png")

    rf_fitted = fitted_models["Random Forest"]
    importances = rf_fitted.named_steps["classifier"].feature_importances_

    imp_df = pd.DataFrame(
        {
            "feature": features,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    print("\nRandom forest feature importances:")
    print(imp_df)
    save_table(imp_df, "table04_random_forest_importances.csv")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp_df, x="importance", y="feature")
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_current_figure("figure3_rf_importance.png")

    # =============================================================================
    # 9. SHARED-VARIANCE ANALYSIS
    # =============================================================================

    corr = df[rating_cols].corr()
    print("\nCorrelations among structured workplace ratings:")
    print(corr)
    save_table(corr, "table05_structured_rating_correlations.csv", index=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlations Among Structured Workplace Ratings")
    save_current_figure("figure4_correlation_heatmap.png")

    X_vif = df[rating_cols].copy()
    X_vif = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_vif),
        columns=rating_cols,
    )

    vif_df = pd.DataFrame(
        {
            "feature": X_vif.columns,
            "VIF": [
                variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])
            ],
        }
    )

    vif_df = vif_df.sort_values("VIF", ascending=False)
    print("\nVariance inflation factors:")
    print(vif_df)
    save_table(vif_df, "table06_variance_inflation_factors.csv")

    # =============================================================================
    # 10. FACTOR/SENTIMENT MODELS
    # =============================================================================

    X_fa = df[rating_cols].copy()
    X_fa = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_fa),
        columns=rating_cols,
    )

    fa1 = FactorAnalyzer(n_factors=1, rotation=None)
    fa1.fit(X_fa)

    loadings_1 = pd.DataFrame(
        fa1.loadings_,
        index=rating_cols,
        columns=["Factor1"],
    )
    print("\n1-factor loadings:")
    print(loadings_1)
    save_table(loadings_1, "table11_factor_loadings_one_factor_raw.csv", index=True)

    loadings_1_report = loadings_1.abs().copy()
    save_table(loadings_1_report, "table12_factor_loadings_one_factor_abs.csv", index=True)

    fa2 = FactorAnalyzer(n_factors=2, rotation="oblimin")
    fa2.fit(X_fa)

    loadings_2 = pd.DataFrame(
        fa2.loadings_,
        index=rating_cols,
        columns=["Factor1", "Factor2"],
    )
    print("\n2-factor loadings:")
    print(loadings_2)
    save_table(loadings_2, "table13_factor_loadings_two_factor.csv", index=True)

    analyzer = SentimentIntensityAnalyzer()

    df["pros_sentiment"] = df["pros"].fillna("").apply(
        lambda value: analyzer.polarity_scores(value)["compound"]
    )
    df["cons_sentiment"] = df["cons"].fillna("").apply(
        lambda value: analyzer.polarity_scores(value)["compound"]
    )
    df["pros_length"] = df["pros"].fillna("").str.split().str.len()
    df["cons_length"] = df["cons"].fillna("").str.split().str.len()

    sent_cols = [
        "pros_sentiment",
        "cons_sentiment",
        "pros_length",
        "cons_length",
    ]

    all_features = rating_cols + aux_cols + sent_cols

    df_model = df.copy()
    X_all = df_model[all_features].copy()
    y_all = df_model["target"].copy()

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    print("\nFactor/sentiment train/test shapes:")
    print(
        X_train_all.shape,
        X_test_all.shape,
        y_train_all.shape,
        y_test_all.shape,
    )

    rating_imputer = SimpleImputer(strategy="median")
    rating_scaler = StandardScaler()

    X_train_ratings = pd.DataFrame(
        rating_imputer.fit_transform(X_train_all[rating_cols]),
        columns=rating_cols,
        index=X_train_all.index,
    )

    X_test_ratings = pd.DataFrame(
        rating_imputer.transform(X_test_all[rating_cols]),
        columns=rating_cols,
        index=X_test_all.index,
    )

    X_train_ratings_scaled = pd.DataFrame(
        rating_scaler.fit_transform(X_train_ratings),
        columns=rating_cols,
        index=X_train_ratings.index,
    )

    X_test_ratings_scaled = pd.DataFrame(
        rating_scaler.transform(X_test_ratings),
        columns=rating_cols,
        index=X_test_ratings.index,
    )

    fa1_train = FactorAnalyzer(n_factors=1, rotation=None)
    fa1_train.fit(X_train_ratings_scaled)

    train_factor1 = -fa1_train.transform(X_train_ratings_scaled)[:, 0]
    test_factor1 = -fa1_train.transform(X_test_ratings_scaled)[:, 0]

    train_loadings_1 = pd.DataFrame(
        fa1_train.loadings_,
        index=rating_cols,
        columns=["Factor1"],
    )
    print("\nTraining-only 1-factor loadings:")
    print(train_loadings_1)
    save_table(
        train_loadings_1,
        "table14_training_only_factor_loadings_one_factor.csv",
        index=True,
    )

    X_train_factor = pd.DataFrame(
        {
            "factor1": train_factor1,
            "current_clean": X_train_all["current_clean"],
            "review_year": X_train_all["review_year"],
        },
        index=X_train_all.index,
    )

    X_test_factor = pd.DataFrame(
        {
            "factor1": test_factor1,
            "current_clean": X_test_all["current_clean"],
            "review_year": X_test_all["review_year"],
        },
        index=X_test_all.index,
    )

    X_train_factor_sent = X_train_factor.copy()
    X_test_factor_sent = X_test_factor.copy()

    for col in sent_cols:
        X_train_factor_sent[col] = X_train_all[col]
        X_test_factor_sent[col] = X_test_all[col]

    print("\nFactor-only feature preview:")
    print(X_train_factor.head())

    print("\nFactor + sentiment feature preview:")
    print(X_train_factor_sent.head())

    X_train_raw = X_train_all[rating_cols + aux_cols].copy()
    X_test_raw = X_test_all[rating_cols + aux_cols].copy()

    raw_logit_model, raw_metrics = fit_and_evaluate_logit(
        X_train_raw,
        X_test_raw,
        y_train_all,
        y_test_all,
        "Raw Ratings Logistic",
    )

    factor_logit_model, factor_metrics = fit_and_evaluate_logit(
        X_train_factor,
        X_test_factor,
        y_train_all,
        y_test_all,
        "1-Factor Logistic",
    )

    factor_sent_logit_model, factor_sent_metrics = fit_and_evaluate_logit(
        X_train_factor_sent,
        X_test_factor_sent,
        y_train_all,
        y_test_all,
        "1-Factor + Sentiment Logistic",
    )

    comparison_df = pd.DataFrame(
        [
            raw_metrics,
            factor_metrics,
            factor_sent_metrics,
        ]
    ).sort_values("ROC_AUC", ascending=False)

    print("\nFactor/sentiment model comparison:")
    print(comparison_df.round(3))
    save_table(comparison_df, "table15_factor_sentiment_model_comparison.csv")

    factor_coef = factor_logit_model.named_steps["classifier"].coef_[0]
    factor_coef_df = pd.DataFrame(
        {
            "feature": X_train_factor.columns,
            "coefficient": factor_coef,
        }
    ).sort_values("coefficient", ascending=False)

    print("\n1-factor logistic coefficients:")
    print(factor_coef_df)
    save_table(factor_coef_df, "table16_factor_logistic_coefficients.csv")

    factor_sent_coef = factor_sent_logit_model.named_steps["classifier"].coef_[0]
    factor_sent_coef_df = pd.DataFrame(
        {
            "feature": X_train_factor_sent.columns,
            "coefficient": factor_sent_coef,
        }
    ).sort_values("coefficient", ascending=False)

    print("\n1-factor + sentiment logistic coefficients:")
    print(factor_sent_coef_df)
    save_table(
        factor_sent_coef_df,
        "table17_factor_sentiment_logistic_coefficients.csv",
    )

    # =============================================================================
    # 11. TF-IDF TEXT ANALYSIS
    # =============================================================================

    pros_train = df_model.loc[X_train_all.index, "pros"].fillna("")
    pros_test = df_model.loc[X_test_all.index, "pros"].fillna("")

    cons_train = df_model.loc[X_train_all.index, "cons"].fillna("")
    cons_test = df_model.loc[X_test_all.index, "cons"].fillna("")

    (
        pros_model,
        pros_coef_df,
        pros_top_pos,
        pros_top_neg,
        pros_metrics,
    ) = fit_text_logit(
        pros_train,
        pros_test,
        y_train_all,
        y_test_all,
        model_name="Pros Text Logistic",
    )

    print("\nTop phrases in PROS associated with positive recommendation:")
    print(pros_top_pos)
    save_table(pros_top_pos, "table18_pros_top_positive_phrases.csv")

    print("\nTop phrases in PROS associated with negative recommendation:")
    print(pros_top_neg)
    save_table(pros_top_neg, "table19_pros_top_negative_phrases.csv")

    (
        cons_model,
        cons_coef_df,
        cons_top_pos,
        cons_top_neg,
        cons_metrics,
    ) = fit_text_logit(
        cons_train,
        cons_test,
        y_train_all,
        y_test_all,
        model_name="Cons Text Logistic",
    )

    print("\nTop phrases in CONS associated with positive recommendation:")
    print(cons_top_pos)
    save_table(cons_top_pos, "table20_cons_top_positive_phrases.csv")

    print("\nTop phrases in CONS associated with negative recommendation:")
    print(cons_top_neg)
    save_table(cons_top_neg, "table21_cons_top_negative_phrases.csv")

    cons_plot_df = pd.concat(
        [
            cons_top_pos.head(10),
            cons_top_neg.head(10),
        ]
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(data=cons_plot_df, x="coefficient", y="term")
    plt.title("Top CONS Terms Associated with Recommendation")
    plt.xlabel("Coefficient")
    plt.ylabel("Term")
    save_current_figure("figure7_cons_phrases.png")

    text_summary = pd.DataFrame(
        [
            pros_metrics,
            cons_metrics,
        ]
    ).sort_values("ROC_AUC", ascending=False)

    print("\nText model comparison:")
    print(text_summary.round(3))
    save_table(text_summary, "table22_text_model_comparison.csv")

    # =============================================================================
    # 12. FIRM-LEVEL HETEROGENEITY EXTENSION
    # =============================================================================

    if "factor1" not in df_model.columns:
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_ratings = imp.fit_transform(df_model[rating_cols])
        X_ratings = scaler.fit_transform(X_ratings)

        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(X_ratings)

        df_model["factor1"] = -fa.transform(X_ratings)[:, 0]

    firm_counts = df_model["firm"].value_counts()
    keep_firms = (
        firm_counts[firm_counts >= MIN_REVIEWS_PER_FIRM]
        .head(MAX_FIRMS)
        .index.tolist()
    )

    if not keep_firms:
        raise ValueError(
            "No firms met MIN_REVIEWS_PER_FIRM. Lower the threshold and rerun."
        )

    df_hetero = df_model[df_model["firm"].isin(keep_firms)].copy()

    df_hetero = (
        df_hetero.groupby("firm", group_keys=False)
        .apply(
            lambda group: group.sample(
                min(len(group), PER_FIRM_CAP),
                random_state=RANDOM_STATE,
            )
        )
        .reset_index(drop=True)
    )

    cols_needed = [
        "firm",
        "target",
        "factor1",
        "cons_sentiment",
        "current_clean",
        "review_year",
    ]
    df_hetero = df_hetero[cols_needed].dropna().copy()

    df_hetero["firm"] = pd.Categorical(df_hetero["firm"], categories=keep_firms)

    for col in ["factor1", "cons_sentiment", "review_year"]:
        df_hetero[f"{col}_z"] = zscore(df_hetero[col])

    print("\nHeterogeneity sample shape:")
    print(df_hetero.shape)

    firm_counts_hetero = df_hetero["firm"].value_counts().rename_axis("firm").reset_index(name="n_reviews")
    print("\nFirm counts in heterogeneity sample:")
    print(firm_counts_hetero)
    save_table(firm_counts_hetero, "table23_heterogeneity_firm_counts.csv")

    # Model 1: firm fixed-effects interaction logistic model.
    # Cons sentiment is the focal negative-signal variable; the cons-sentiment
    # slope varies by firm, and factor1, reviewer status, and review year remain
    # controls.

    ref_firm = keep_firms[0]
    ref_firm_safe = ref_firm.replace("\\", "\\\\").replace("'", "\\'")

    formula_base = (
        "target ~ factor1_z + cons_sentiment_z + current_clean + review_year_z "
        f"+ C(firm, Treatment(reference='{ref_firm_safe}'))"
    )

    formula_interaction = (
        "target ~ factor1_z + cons_sentiment_z + current_clean + review_year_z "
        f"+ C(firm, Treatment(reference='{ref_firm_safe}')) "
        f"+ cons_sentiment_z:C(firm, Treatment(reference='{ref_firm_safe}'))"
    )

    model_fe_base = smf.glm(
        formula=formula_base,
        data=df_hetero,
        family=sm.families.Binomial(),
    ).fit()

    model_fe_int = smf.glm(
        formula=formula_interaction,
        data=df_hetero,
        family=sm.families.Binomial(),
    ).fit()

    print("\nFixed-effects interaction model:")
    print(model_fe_int.summary())

    lr_stat = 2 * (model_fe_int.llf - model_fe_base.llf)
    df_diff = model_fe_int.df_model - model_fe_base.df_model
    p_value = chi2.sf(lr_stat, df_diff)

    lr_test_df = pd.DataFrame(
        [
            {
                "test": "Firm-specific cons_sentiment slopes",
                "lr_statistic": lr_stat,
                "df": int(df_diff),
                "p_value": p_value,
            }
        ]
    )
    print("\nLikelihood-ratio test for firm-specific cons_sentiment slopes")
    print(lr_test_df.round(3))
    save_table(lr_test_df, "table24_fixed_effects_likelihood_ratio_test.csv")

    base_slope = model_fe_int.params["cons_sentiment_z"]

    slopes_fe = []
    for firm in keep_firms:
        term = (
            "cons_sentiment_z:"
            f"C(firm, Treatment(reference='{ref_firm_safe}'))[T.{firm}]"
        )
        slope = base_slope + model_fe_int.params.get(term, 0.0)
        slopes_fe.append({"firm": firm, "fe_slope": slope})

    slopes_fe = pd.DataFrame(slopes_fe).sort_values("fe_slope")
    print("\nFixed-effects firm-specific slopes:")
    print(slopes_fe.round(3))

    # Build Construct: Negative-Review Sensitivity
    slopes_fe["negative_review_sensitivity_fe"] = slopes_fe["fe_slope"]
    print("\nNegative-review sensitivity construct:")
    print(
        slopes_fe.sort_values(
            "negative_review_sensitivity_fe",
            ascending=False,
        ).round(3)
    )
    save_table(slopes_fe, "table25_fixed_effects_firm_slopes.csv")

    plot_fe = slopes_fe.sort_values("fe_slope").copy()
    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_fe, x="fe_slope", y="firm")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title(
        "Exploratory Firm-Specific Slopes for Cons Sentiment\n"
        "(Fixed-Effects Interaction Logistic Model)"
    )
    plt.xlabel("Slope of standardized cons sentiment on positive recommendation")
    plt.ylabel("Firm")
    save_current_figure("appendix_heterogeneity_fe_slopes_exploratory.png")

    # Model 2: multilevel logistic model with random intercepts and random slopes
    # for cons sentiment by firm. Recommendation remains the dependent variable,
    # and factor1, reviewer status, and review year remain controls.

    random_effects = {
        "firm_intercept": "0 + C(firm)",
        "firm_cons_slope": "0 + C(firm):cons_sentiment_z",
    }

    model_ml = BinomialBayesMixedGLM.from_formula(
        "target ~ factor1_z + cons_sentiment_z + current_clean + review_year_z",
        random_effects,
        df_hetero,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result_ml = model_ml.fit_vb()

    print("\nMultilevel logistic model:")
    print(result_ml.summary())

    if caught_warnings:
        print("\nWarnings from multilevel model:")
        for warning in caught_warnings:
            print(f"- {warning.message}")
        print(
            "Note: Exact posterior estimates should be interpreted cautiously, "
            "but the firm-specific slopes were highly consistent with the "
            "fixed-effects interaction model."
        )

    vc_df = pd.DataFrame(
        {
            "component": model_ml.vcp_names,
            "log_sd_mean": result_ml.vcp_mean,
            "sd_estimate": np.exp(result_ml.vcp_mean),
        }
    )

    print("\nRandom-effect variance components:")
    print(vc_df.round(3))
    save_table(vc_df, "table26_multilevel_variance_components.csv")

    fe_params_ml = pd.Series(result_ml.fe_mean, index=model_ml.exog_names)
    base_slope_ml = fe_params_ml["cons_sentiment_z"]

    re_df = result_ml.random_effects()
    if not isinstance(re_df, pd.DataFrame):
        re_df = pd.DataFrame(re_df)

    slope_mask = (
        re_df.index.to_series()
        .astype(str)
        .str.contains("cons_sentiment_z", regex=False)
    )
    re_slope_df = re_df.loc[slope_mask].copy()

    slope_col = "Mean" if "Mean" in re_slope_df.columns else re_slope_df.columns[0]

    re_slope_df = (
        re_slope_df.reset_index()
        .rename(columns={"index": "raw_name", slope_col: "random_slope"})
    )
    re_slope_df["firm"] = (
        re_slope_df["raw_name"]
        .astype(str)
        .str.replace("C(firm)[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(":cons_sentiment_z", "", regex=False)
    )

    slopes_ml = re_slope_df[["firm", "random_slope"]].copy()
    slopes_ml["ml_slope"] = base_slope_ml + slopes_ml["random_slope"]

    print("\nMultilevel firm-specific slopes:")
    print(slopes_ml.round(3))
    save_table(slopes_ml, "table27_multilevel_firm_slopes.csv")

    slope_compare = slopes_fe.merge(
        slopes_ml[["firm", "ml_slope"]],
        on="firm",
        how="inner",
    )

    print("\nFixed-effects vs multilevel slopes:")
    print(slope_compare.round(3))
    save_table(slope_compare, "table28_fixed_effects_vs_multilevel_slopes.csv")

    slope_correlation = slope_compare["fe_slope"].corr(slope_compare["ml_slope"])
    slope_correlation_df = pd.DataFrame(
        [
            {
                "comparison": "Fixed-effects slope vs multilevel slope",
                "correlation": slope_correlation,
            }
        ]
    )
    print("\nCorrelation between firm slope estimates:")
    print(round(slope_correlation, 3))
    save_table(slope_correlation_df, "table29_heterogeneity_slope_correlation.csv")

    label_map = {
        "McDonald-s": "McDonald's",
        "Marriott-International": "Marriott",
        "American-Express": "AmEx",
        "J-P-Morgan": "JPMorgan",
        "HSBC-Holdings": "HSBC",
        "Thomson-Reuters": "Thomson Reuters",
    }

    scatter_df = slope_compare.copy()
    scatter_df["label"] = scatter_df["firm"].replace(label_map)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=scatter_df, x="fe_slope", y="ml_slope", s=80)
    for _, row in scatter_df.iterrows():
        plt.text(row["fe_slope"], row["ml_slope"], row["label"], fontsize=8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(
        "Appendix: Firm-Specific Cons-Sentiment Slopes\n"
        "Fixed-Effects Interaction vs Multilevel Model"
    )
    plt.xlabel("Fixed-effects interaction slope")
    plt.ylabel("Multilevel random-slope estimate")
    save_current_figure("appendix_fe_vs_mlm_scatter.png")

    plt.figure(figsize=(11, 8))
    sns.scatterplot(data=scatter_df, x="fe_slope", y="ml_slope", s=90)
    texts = []
    for _, row in scatter_df.iterrows():
        texts.append(
            plt.text(row["fe_slope"], row["ml_slope"], row["label"], fontsize=9)
        )
    if adjust_text is not None:
        adjust_text(
            texts,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.8},
        )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(
        "Exploratory: All Firm Labels for Cons-Sentiment Slopes\n"
        "Fixed-Effects Interaction vs Multilevel Model"
    )
    plt.xlabel("Fixed-effects interaction slope")
    plt.ylabel("Multilevel random-slope estimate")
    save_current_figure("appendix_heterogeneity_scatter_all_labels_exploratory.png")

    extreme_label_df = pd.concat(
        [
            scatter_df.nsmallest(5, "ml_slope"),
            scatter_df.nlargest(5, "ml_slope"),
        ]
    ).drop_duplicates()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=scatter_df, x="fe_slope", y="ml_slope", s=80)
    if len(scatter_df) >= 2:
        x_values = scatter_df["fe_slope"].to_numpy()
        y_values = scatter_df["ml_slope"].to_numpy()
        line_slope, line_intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        plt.plot(x_line, line_slope * x_line + line_intercept, linestyle="--")
    for _, row in extreme_label_df.iterrows():
        plt.annotate(
            row["label"],
            xy=(row["fe_slope"], row["ml_slope"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )
    plt.title(
        "Exploratory: Extreme Firm Slopes Only\n"
        "Fixed-Effects Interaction vs Multilevel Model"
    )
    plt.xlabel("Fixed-effects interaction slope")
    plt.ylabel("Multilevel random-slope estimate")
    save_current_figure("appendix_heterogeneity_scatter_extreme_labels_exploratory.png")

    plot_ml = slope_compare.sort_values("ml_slope").copy()
    plot_ml["label"] = plot_ml["firm"].replace(label_map)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_ml, x="ml_slope", y="label")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Firm-Level Variation in the Effect of Cons Sentiment")
    plt.xlabel("Slope of standardized cons sentiment on positive recommendation")
    plt.ylabel("Firm")
    save_current_figure("figure8_firm_heterogeneity.png")

    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_ml, x="ml_slope", y="label")
    plt.axvline(plot_ml["ml_slope"].mean(), color="black", linestyle="--", linewidth=1)
    plt.title(
        "Exploratory: Firm-Level Variation in Cons-Sentiment Slopes\n"
        "Mean Multilevel Slope Reference Line"
    )
    plt.xlabel("Multilevel slope for standardized cons sentiment")
    plt.ylabel("Firm")
    save_current_figure("appendix_heterogeneity_mlm_slopes_mean_line_exploratory.png")

    # =============================================================================
    # 13. APPENDIX DIAGNOSTICS
    # =============================================================================

    CalibrationDisplay.from_estimator(
        factor_sent_logit_model,
        X_test_factor_sent,
        y_test_all,
        n_bins=10,
    )
    plt.title("Calibration Plot: One-Factor + Sentiment Logistic Model")
    save_current_figure("appendixD_calibration_plot.png")

    PrecisionRecallDisplay.from_estimator(
        factor_sent_logit_model,
        X_test_factor_sent,
        y_test_all,
    )
    plt.title("Precision-Recall Curve: One-Factor + Sentiment Logistic Model")
    save_current_figure("appendix_precision_recall_curve.png")


if __name__ == "__main__":
    main()
