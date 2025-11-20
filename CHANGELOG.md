# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-11-20

### Added
- **DUAL-NAME ALIAS SYSTEM**: Every function now has TWO names!
  - Full descriptive name (e.g., `k_fold_cross_validation`)
  - Abbreviated alias (e.g., `kfold`)
  - Use whichever style you prefer - both work identically!

- **ML Module - Cross-Validation**: 8 powerful CV functions (16 names total with aliases)
  - `k_fold_cross_validation()` / `kfold()` - Standard K-fold CV
  - `stratified_k_fold()` / `skfold()` - Stratified K-fold for imbalanced data
  - `time_series_split()` / `tssplit()` - Time series CV (no data leakage)
  - `leave_one_out_cv()` / `loocv()` - Leave-one-out CV
  - `shuffle_split_cv()` / `shuffle_cv()` - Random shuffle split
  - `cross_validate_score()` / `cv_score()` - Automated CV scoring
  - `holdout_validation_split()` / `holdout()` - Simple train/test split
  - `train_val_test_split()` / `tvt_split()` - Three-way split

- **ML Module - Hyperparameter Tuning**: 10 tuning functions (20 names total with aliases)
  - `grid_search_cv()` / `gridsearch()` - Exhaustive grid search
  - `random_search_cv()` / `randomsearch()` - Random parameter search
  - `bayesian_search_simple()` / `bayesopt()` - Bayesian optimization
  - `generate_param_grid()` / `param_grid()` - Generate parameter grids
  - `extract_best_params()` / `best_params()` - Extract best parameters
  - `format_cv_results()` / `cv_results()` - Format CV results
  - `learning_curve_data()` / `learning_curve()` - Learning curve analysis
  - `validation_curve_data()` / `val_curve()` - Validation curve analysis
  - `early_stopping_monitor()` / `early_stop()` - Early stopping utility
  - `compare_models_cv()` / `compare_models()` - Compare multiple models

- **ML Module - Metric Aliases**: Added short aliases for regression metrics
  - `mean_absolute_error()` / `mae()`
  - `mean_squared_error()` / `mse()`
  - `root_mean_squared_error()` / `rmse()`

### Changed
- All ML functions now support dual-name system
- Documentation updated to show both naming styles
- Examples demonstrate both full and abbreviated names

### Notes
- **18 NEW functions** (36 names with aliases!)
- Total ML functions: 29 (58 names with aliases!)
- No breaking changes - all existing code works
- Choose your preferred naming style

## [0.1.4] - 2025-11-18

### Added
- **NEW ML Module**: Complete Model Evaluation Metrics toolkit with 11 essential functions
  - `accuracy_score()` - Calculate classification accuracy
  - `precision_score()` - Measure precision (avoid false positives)
  - `recall_score()` - Measure recall/sensitivity (avoid false negatives)
  - `f1_score()` - Harmonic mean of precision and recall
  - `confusion_matrix()` - Generate confusion matrix for classification
  - `classification_report()` - Comprehensive classification metrics report
  - `mean_squared_error()` - MSE for regression problems
  - `mean_absolute_error()` - MAE for regression problems
  - `root_mean_squared_error()` - RMSE for regression problems
  - `r2_score()` - R-squared coefficient of determination
  - `roc_auc_score()` - ROC AUC score for binary classification
  - No scikit-learn required for basic metrics
  - Perfect for model evaluation without heavy dependencies
  - Comprehensive documentation with real-world examples

## [0.1.3] - 2025-11-17

### Added
- **Data Module**: Complete Feature Engineering toolkit with 7 powerful functions
  - `create_polynomial_features()` - Generate polynomial features (x, x^2, x^3) for non-linear relationships
  - `bin_numerical_feature()` - Convert continuous data into categorical bins/groups
  - `one_hot_encode()` - One-hot encoding for categorical variables
  - `label_encode()` - Label encoding for ordinal categories
  - `extract_datetime_features()` - Extract temporal features (hour, day, month, is_weekend) from timestamps
  - `handle_missing_values()` - Multiple strategies for missing data (mean, median, forward/backward fill)
  - `create_interaction_features()` - Create feature interactions (multiply, add, subtract, divide)
  - All functions work without external ML libraries
  - Comprehensive documentation with real-world examples
  - Perfect for ML preprocessing pipelines

## [0.1.2] - 2025-11-16

### Added
- **AI Module**: `similarity_search()` - Find similar documents using TF-IDF, Jaccard, Levenshtein, or N-gram methods
  - Multiple similarity algorithms (tfidf, jaccard, levenshtein, ngram)
  - No external dependencies or API calls
  - Perfect for search functionality and document retrieval
  - Works offline with fast performance
- **AI Module**: `cosine_similarity()` - Calculate cosine similarity between vectors
- **Data Module**: `train_test_split()` - Split data into train and test sets
  - Supports stratified splitting for balanced classes
  - Random seed for reproducibility
  - Handles both features and labels
  - Common split ratios (70-30, 80-20, 60-20-20)
- **Data Module**: `normalize_data()` - Min-max normalization to [0, 1]
- **Data Module**: `standardize_data()` - Z-score standardization (mean=0, std=1)

## [0.1.1] - 2025-11-15

### Added
- **AI Module**: `token_counter()` - Smart LLM token estimation function
  - Supports multiple models (GPT-3.5, GPT-4, Claude, Llama, Gemini)
  - Accurate token counting without API calls
  - Cost estimation for different models
  - Detailed breakdown mode with character/word counts
  - Works offline - no external dependencies

## [0.1.0] - 2025-11-11

### Added
- Initial library structure
- Core module setup (ai, data, files, text, image, audio, web, security, database, datetime, validation, conversion, automation, utils)
- Basic package configuration
- README with comprehensive documentation
- MIT License
- Contributing guidelines
