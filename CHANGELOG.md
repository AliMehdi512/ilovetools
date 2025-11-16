# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
