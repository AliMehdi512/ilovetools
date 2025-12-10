# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2025-12-10

### Fixed
- Corrected packaging metadata and synchronized versions across `setup.py`, `pyproject.toml`, and `ilovetools/__init__.py` so builds and CI produce consistent artifacts.
- Fixed deprecated/incorrect metadata fields in `pyproject.toml` (license/metadata formatting) to satisfy modern build tools.

### Changed
- Updated project metadata and dependency lists for consistent installation behavior.

### Notes
- This release (0.2.4) is intended to follow the published 0.2.3 release on PyPI and correct packaging issues that prevented a reliable publish.

## [0.2.5] - 2025-12-10

### Fixed
- Bumped package version to 0.2.5 and republished to avoid PyPI filename/hash reuse protection that blocked re-uploads of 0.2.4.

### Notes
- Release created to unblock publishing; see 0.2.4 for packaging fixes and metadata synchronization.

## [0.2.9] - 2025-12-10

### Fixed
- Bumped package version to 0.2.9 to avoid PyPI filename/hash reuse protection and successfully publish corrected artifacts.

### Notes
- Release created to get a successful publish after filename-reuse blocks prevented re-uploading previous artifact names.

## [0.2.9.post1] - 2025-12-10

### Fixed
- Publish as a PEP 440 post-release `0.2.9.post1` to generate a unique artifact filename and avoid PyPI's filename/hash reuse protection.

### Notes
- `pyproject.toml` is now the authoritative source for dependency metadata; `setup.py` no longer contains `install_requires`/`extras_require` to avoid build-time warnings.

## [0.2.6] - 2025-12-10

### Fixed
- **Version Synchronization**: Fixed version inconsistencies across setup.py, pyproject.toml, and __init__.py
- **Package Configuration**: Enhanced pyproject.toml with complete dependency specifications
- All version files now consistently report 0.2.6

### Changed
- Improved package build configuration for better PyPI compatibility
- Updated project metadata for consistency

## [0.2.3] - 2025-11-30

### Added
- **ML Module - Clustering Algorithms**: 15 powerful clustering functions (30 names with aliases)
  - `kmeans_clustering()` / `kmeans()` - K-Means clustering algorithm
  - `hierarchical_clustering()` / `hierarchical()` - Hierarchical agglomerative clustering
  - `dbscan_clustering()` / `dbscan()` - DBSCAN density-based clustering
  - `elbow_method()` / `elbow()` - Find optimal K using elbow method
  - `silhouette_score()` / `silhouette()` - Calculate silhouette score
  - `euclidean_distance()` / `euclidean()` - Euclidean distance metric
  - `manhattan_distance()` / `manhattan()` - Manhattan distance metric
  - `cosine_similarity_distance()` / `cosine_dist()` - Cosine distance metric
  - `initialize_centroids()` / `init_centroids()` - Initialize cluster centroids
  - `assign_clusters()` / `assign()` - Assign points to clusters
  - `update_centroids()` / `update()` - Update cluster centroids
  - `calculate_inertia()` / `inertia()` - Calculate within-cluster sum of squares
  - `dendrogram_data()` / `dendrogram()` - Generate dendrogram data
  - `cluster_purity()` / `purity()` - Calculate cluster purity score
  - `davies_bouldin_index()` / `davies_bouldin()` - Davies-Bouldin index

### Notes
- **15 NEW functions** (30 names with aliases!)
- Total ML functions: 134 (268 names with aliases!)
- Implements K-Means, Hierarchical, DBSCAN clustering
- Distance metrics: Euclidean, Manhattan, Cosine
- Cluster validation: Silhouette, Elbow, Davies-Bouldin, Purity
- All functions follow dual-name alias system
- No breaking changes

## [0.2.2] - 2025-11-29

### Added
- **ML Module - Time Series Analysis**: 15 powerful time series functions (30 names with aliases)
  - `moving_average()` / `ma()` - Simple Moving Average (SMA)
  - `exponential_moving_average()` / `ema()` - Exponential Moving Average
  - `weighted_moving_average()` / `wma()` - Weighted Moving Average
  - `seasonal_decompose()` / `decompose()` - Seasonal decomposition
  - `difference_series()` / `diff()` - Differencing for stationarity
  - `autocorrelation()` / `acf()` - Autocorrelation function
  - `partial_autocorrelation()` / `pacf()` - Partial autocorrelation
  - `detect_trend()` / `trend()` - Trend detection
  - `detect_seasonality()` / `seasonality()` - Seasonality detection
  - `remove_trend()` / `detrend()` - Remove trend
  - `remove_seasonality()` / `deseasonalize()` - Remove seasonality
  - `rolling_statistics()` / `rolling_stats()` - Rolling statistics
  - `lag_features()` / `lag()` - Create lag features
  - `time_series_split_cv()` / `ts_cv()` - Time series cross-validation
  - `forecast_accuracy()` / `forecast_acc()` - Forecast accuracy metrics

### Notes
- **15 NEW functions** (30 names with aliases!)
- Total ML functions: 119 (238 names with aliases!)
- Implements moving averages, seasonal decomposition, ACF/PACF, trend/seasonality detection
- All functions follow dual-name alias system
- No breaking changes

## [0.2.1] - 2025-11-28

### Added
- **ML Module - Dimensionality Reduction**: 15 powerful dimensionality reduction functions (30 names with aliases)
  - `pca_transform()` / `pca()` - Principal Component Analysis
  - `explained_variance_ratio()` / `exp_var()` - Explained variance ratios
  - `scree_plot_data()` / `scree_plot()` - Scree plot data generation
  - `cumulative_variance()` / `cum_var()` - Cumulative variance calculation
  - `pca_inverse_transform()` / `pca_inverse()` - Inverse PCA transformation
  - `truncated_svd()` / `svd()` - Truncated SVD decomposition
  - `kernel_pca_transform()` / `kpca()` - Kernel PCA (non-linear)
  - `incremental_pca_transform()` / `ipca()` - Incremental PCA for large datasets
  - `feature_projection()` / `project()` - Generic feature projection
  - `dimensionality_reduction_ratio()` / `dim_ratio()` - Reduction ratio calculation
  - `reconstruction_error()` / `recon_error()` - Reconstruction error metrics
  - `optimal_components()` / `opt_components()` - Find optimal number of components
  - `whitening_transform()` / `whiten()` - Whitening transformation
  - `component_loadings()` / `loadings()` - Component loading calculation
  - `biplot_data()` / `biplot()` - Biplot data generation

### Notes
- **15 NEW functions** (30 names with aliases!)
- Total ML functions: 104 (208 names with aliases!)
- Implements PCA, SVD, Kernel PCA, Incremental PCA
- All functions follow dual-name alias system
- No breaking changes

## [0.2.0] - 2025-11-26

### Added
- **ML Module - Imbalanced Data Handling**: 12 powerful imbalanced data functions (24 names with aliases)
  - `random_oversampling()` / `random_oversample()` - Random oversampling
  - `random_undersampling()` / `random_undersample()` - Random undersampling
  - `smote_oversampling()` / `smote()` - SMOTE synthetic sampling
  - `tomek_links_undersampling()` / `tomek_links()` - Tomek links cleaning
  - `class_weight_calculator()` / `class_weights()` - Calculate class weights
  - `stratified_sampling()` / `stratified_sample()` - Stratified sampling
  - `compute_class_distribution()` / `class_dist()` - Class distribution stats
  - `balance_dataset()` / `balance_data()` - Unified balancing interface
  - `minority_class_identifier()` / `minority_class()` - Identify minority class
  - `imbalance_ratio()` / `imbalance_ratio_alias()` - Calculate imbalance ratio
  - `synthetic_sample_generator()` / `synthetic_sample()` - Generate synthetic samples
  - `near_miss_undersampling()` / `near_miss()` - NearMiss undersampling

### Notes
- **12 NEW functions** (24 names with aliases!)
- Total ML functions: 89 (178 names with aliases!)
- Implements SMOTE, Tomek links, NearMiss, class weights
- All functions follow dual-name alias system
- No breaking changes
- Major version bump to 0.2.0

## [0.1.9] - 2025-11-25

### Added
- **ML Module - Pipeline Utilities**: 12 powerful pipeline functions (24 names with aliases)
  - `create_pipeline()` / `create_pipe()` - Create new ML pipeline
  - `add_pipeline_step()` / `add_step()` - Add step to pipeline
  - `execute_pipeline()` / `execute_pipe()` - Execute pipeline on data
  - `validate_pipeline()` / `validate_pipe()` - Validate pipeline structure
  - `serialize_pipeline()` / `serialize_pipe()` - Serialize to JSON
  - `deserialize_pipeline()` / `deserialize_pipe()` - Deserialize from JSON
  - `pipeline_transform()` / `pipe_transform()` - Transform with fitted pipeline
  - `pipeline_fit_transform()` / `pipe_fit_transform()` - Fit and transform
  - `get_pipeline_params()` / `get_params()` - Get all parameters
  - `set_pipeline_params()` / `set_params()` - Set parameters
  - `clone_pipeline()` / `clone_pipe()` - Clone pipeline
  - `pipeline_summary()` / `pipe_summary()` - Get summary statistics

### Notes
- **12 NEW functions** (24 names with aliases!)
- Total ML functions: 77 (154 names with aliases!)
- Implements pipeline creation, execution, serialization
- All functions follow dual-name alias system
- No breaking changes

## [0.1.8] - 2025-11-23

### Added
- **ML Module - Model Interpretation**: 12 powerful interpretation functions (24 names with aliases)
  - `feature_importance_scores()` / `feat_importance_scores()` - Format importance scores
  - `permutation_importance()` / `perm_importance()` - Permutation-based importance
  - `partial_dependence()` / `pdp()` - Partial dependence plots
  - `shap_values_approximation()` / `shap_approx()` - SHAP value approximation
  - `lime_explanation()` / `lime_explain()` - LIME local explanations
  - `decision_path_explanation()` / `decision_path()` - Decision tree path
  - `model_coefficients_interpretation()` / `coef_interpret()` - Linear model coefficients
  - `prediction_breakdown()` / `pred_breakdown()` - Break down predictions
  - `feature_contribution_analysis()` / `feat_contrib()` - Contribution analysis
  - `global_feature_importance()` / `global_importance()` - Global importance ranking
  - `local_feature_importance()` / `local_importance()` - Local importance for instances
  - `model_summary_statistics()` / `model_summary()` - Model summary stats

### Notes
- **12 NEW functions** (24 names with aliases!)
- Total ML functions: 65 (130 names with aliases!)
- Implements SHAP, LIME, permutation importance, PDP
- All functions follow dual-name alias system
- No breaking changes

## [0.1.7] - 2025-11-22

### Added
- **ML Module - Feature Selection**: 12 powerful feature selection functions (24 names with aliases)
  - `correlation_filter()` / `corr_filter()` - Remove highly correlated features
  - `variance_threshold_filter()` / `var_filter()` - Remove low-variance features
  - `chi_square_filter()` / `chi2_filter()` - Chi-square test for categorical features
  - `mutual_information_filter()` / `mi_filter()` - Mutual information scoring
  - `recursive_feature_elimination()` / `rfe()` - RFE wrapper method
  - `forward_feature_selection()` / `forward_select()` - Forward selection
  - `backward_feature_elimination()` / `backward_select()` - Backward elimination
  - `feature_importance_ranking()` / `feat_importance()` - Rank by importance scores
  - `l1_feature_selection()` / `l1_select()` - L1 regularization (Lasso)
  - `univariate_feature_selection()` / `univariate_select()` - Univariate tests
  - `select_k_best_features()` / `select_k_best()` - Automatic k-best selection
  - `remove_correlated_features()` / `remove_corr()` - Remove correlations with details

### Notes
- **12 NEW functions** (24 names with aliases!)
- Total ML functions: 53 (106 names with aliases!)
- Implements filter, wrapper, and embedded methods
- All functions follow dual-name alias system
- No breaking changes

## [0.1.6] - 2025-11-21

### Added
- **ML Module - Ensemble Methods**: 12 powerful ensemble functions (24 names with aliases)
  - `voting_classifier()` / `vote_clf()` - Combine classifiers with voting
  - `voting_regressor()` / `vote_reg()` - Combine regressors with averaging
  - `bagging_predictions()` / `bagging()` - Bootstrap aggregating ensemble
  - `boosting_sequential()` / `boosting()` - Sequential boosting ensemble
  - `stacking_ensemble()` / `stacking()` - Stacking with meta-model
  - `weighted_average_ensemble()` / `weighted_avg()` - Weighted predictions
  - `majority_vote()` / `hard_vote()` - Hard voting for classification
  - `soft_vote()` / `soft_vote_alias()` - Soft voting with probabilities
  - `bootstrap_sample()` / `bootstrap()` - Create bootstrap samples
  - `out_of_bag_score()` / `oob_score()` - OOB validation score
  - `ensemble_diversity()` / `diversity()` - Measure model diversity
  - `blend_predictions()` / `blend()` - Blend predictions with holdout

### Notes
- **12 NEW functions** (24 names with aliases!)
- Total ML functions: 41 (82 names with aliases!)
- Implements bagging, boosting, and stacking
- All functions follow dual-name alias system
- No breaking changes

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
