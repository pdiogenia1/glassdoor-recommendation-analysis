# Table Manifest

The script saves the following CSV tables to the directory supplied with `--table-dir`.

| Filename | Section | Paper use | Description |
|---|---|---|---|
| table01_cross_validation_results.csv | Cross-validation | Model validation table | Mean 5-fold stratified cross-validation metrics for dummy, logistic-regression, and random-forest models. |
| table02_main_model_comparison.csv | Main model evaluation | Main model comparison | Holdout test-set accuracy, precision, recall, F1, and ROC-AUC for the main models. |
| table03_logistic_coefficients.csv | Model interpretation | Coefficient table | Logistic-regression coefficients for the main structured-feature model. |
| table04_random_forest_importances.csv | Model interpretation | Feature-importance table | Random-forest feature importances for the main structured-feature model. |
| table05_structured_rating_correlations.csv | Shared-variance analysis | Correlation table | Correlation matrix for structured workplace rating variables. |
| table06_variance_inflation_factors.csv | Shared-variance analysis | Appendix / diagnostics | Variance inflation factors for structured workplace rating variables. |
| table11_factor_loadings_one_factor_raw.csv | Factor/sentiment models | Measurement diagnostics | Raw one-factor loadings for the structured workplace rating variables. |
| table12_factor_loadings_one_factor_abs.csv | Factor/sentiment models | Report factor-loading table | Absolute-value one-factor loadings for reporting, since factor signs are arbitrary. |
| table13_factor_loadings_two_factor.csv | Factor/sentiment models | Measurement diagnostics | Two-factor oblimin-rotated loadings for structured workplace rating variables. |
| table14_training_only_factor_loadings_one_factor.csv | Factor/sentiment models | Predictive-model diagnostics | One-factor loadings estimated on the training set only for predictive factor-score construction. |
| table15_factor_sentiment_model_comparison.csv | Factor/sentiment models | Factor/sentiment model comparison | Holdout metrics for raw-ratings, one-factor, and one-factor plus sentiment logistic models. |
| table16_factor_logistic_coefficients.csv | Factor/sentiment models | Coefficient table | Coefficients for the one-factor logistic model. |
| table17_factor_sentiment_logistic_coefficients.csv | Factor/sentiment models | Coefficient table | Coefficients for the one-factor plus sentiment logistic model. |
| table18_pros_top_positive_phrases.csv | TF-IDF text analysis | Text-analysis appendix | Top pros-text phrases associated with positive recommendation. |
| table19_pros_top_negative_phrases.csv | TF-IDF text analysis | Text-analysis appendix | Top pros-text phrases associated with negative recommendation. |
| table20_cons_top_positive_phrases.csv | TF-IDF text analysis | Text-analysis appendix | Top cons-text phrases associated with positive recommendation. |
| table21_cons_top_negative_phrases.csv | TF-IDF text analysis | Text-analysis appendix | Top cons-text phrases associated with negative recommendation. |
| table22_text_model_comparison.csv | TF-IDF text analysis | Text-model comparison | Holdout metrics for pros-text and cons-text TF-IDF logistic models. |
| table23_heterogeneity_firm_counts.csv | Firm-level heterogeneity extension | Heterogeneity sample description | Number of reviews per retained firm in the heterogeneity sample. |
| table24_fixed_effects_likelihood_ratio_test.csv | Firm-level heterogeneity extension | Heterogeneity model test | Likelihood-ratio test comparing fixed-effects base and firm-specific cons-sentiment slope models. |
| table25_fixed_effects_firm_slopes.csv | Firm-level heterogeneity extension | Heterogeneity slope table | Firm-specific cons-sentiment slopes from the fixed-effects interaction model, including the negative-review sensitivity construct. |
| table26_multilevel_variance_components.csv | Firm-level heterogeneity extension | Multilevel model diagnostics | Estimated random-effect variance components from the multilevel logistic model. |
| table27_multilevel_firm_slopes.csv | Firm-level heterogeneity extension | Heterogeneity slope table | Firm-specific cons-sentiment slopes from the multilevel logistic model. |
| table28_fixed_effects_vs_multilevel_slopes.csv | Firm-level heterogeneity extension | Model-comparison appendix | Matched fixed-effects and multilevel firm-specific cons-sentiment slopes. |
| table29_heterogeneity_slope_correlation.csv | Firm-level heterogeneity extension | Model-comparison appendix | Correlation between fixed-effects and multilevel firm-specific slope estimates. |
| appendix_learning_curve_logistic_regression.csv | Appendix diagnostics | Learning-curve appendix | Training and cross-validation ROC-AUC values for the logistic-regression learning curve. |
| appendix_learning_curve_random_forest.csv | Appendix diagnostics | Learning-curve appendix | Training and cross-validation ROC-AUC values for the random-forest learning curve. |
