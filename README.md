<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-green?logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<h1 align="center">🎮 Game Prediction ML</h1>

<p align="center">
  A two-phase machine learning project that predicts video game success on Steam — first by <b>regressing</b> on recommendation counts (Milestone 1), then by <b>classifying</b> game popularity into Low / Medium / High tiers (Milestone 2).
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
  - [Phase 1 — Shared Preprocessing Steps](#phase-1--shared-preprocessing-steps)
  - [Phase 2 — Milestone 2 Additions](#phase-2--milestone-2-additions)
- [Feature Engineering](#-feature-engineering)
- [Feature Selection](#-feature-selection)
  - [Regression Feature Selection](#regression-feature-selection)
  - [Classification Feature Selection](#classification-feature-selection)
- [Regression Phase (Milestone 1)](#-regression-phase-milestone-1)
  - [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
  - [Regression Results Comparison](#regression-results-comparison)
- [Classification Phase (Milestone 2)](#-classification-phase-milestone-2)
  - [Target Variable](#target-variable)
  - [Class Imbalance — SMOTE](#class-imbalance--smote)
  - [Models & Hyperparameter Tuning](#models--hyperparameter-tuning-1)
  - [Classification Results Comparison](#classification-results-comparison)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)

---

## 🔍 Project Overview

This project applies a full machine learning pipeline to a Steam games dataset to answer two prediction problems:

| Phase | Task | Target | Notebook |
|---|---|---|---|
| Milestone 1 | **Regression** | `RecommendationCount` (continuous) | `Regression_Models.ipynb` |
| Milestone 2 | **Classification** | `GamePopularity` (Low / Medium / High) | `Classification_Models_Milestone2.ipynb` |

Both milestones share the same preprocessing foundation. Milestone 2 extends it with an additional three-step feature selection pipeline tailored for classification.

---

## 📁 Repository Structure

```
Game-Prediction-ML/
│
├── Data/
│   ├── Data_Preprocessing.ipynb           # Milestone 1 preprocessing
│   ├── Data_Preprocessing_milestone2.ipynb # Milestone 2 preprocessing (+ feature selection)
│   ├── EDA.ipynb                           # Exploratory data analysis
│   ├── data_Extended.ipynb                 # Extended data collection
│   ├── data_preprocessed.csv              # Cleaned dataset
│   ├── data_urls_final.csv                # Raw dataset with URLs
│   ├── dataset_API.csv                    # API-fetched data
│   ├── idlist.csv                         # Game ID list
│   ├── train_data.csv                     # Training data (Milestone 1)
│   └── train_data_2.csv                   # Training data with GamePopularity (Milestone 2)
│
├── Classification_Models_Milestone2.ipynb  # Final classification models
├── Regression_Models.ipynb                 # Final regression models
├── Milestone1_Report.pdf                   # Regression milestone report
├── Milestone2_Report_CHP2.pdf              # Classification milestone report
└── README.md
```

---

## 📊 Dataset

The dataset is sourced from the Steam platform API and contains **11,357 rows × 88 columns** across numerical, categorical, and boolean feature types.

| Feature Type | Count |
|---|---|
| Numerical | ~30 |
| Categorical | ~40 |
| Boolean | ~18 |

**Target Variables:**
- **Regression:** `RecommendationCount` — raw integer count of user recommendations per game.
- **Classification:** `GamePopularity` — an ordinal label (`Low`, `Medium`, `High`) derived from game engagement metrics.

---

## 🧹 Data Preprocessing

Both milestones share the same preprocessing pipeline. The steps below are applied identically in both notebooks. Milestone 2 adds feature selection on top of this shared foundation.

### Phase 1 — Shared Preprocessing Steps

#### Step 1 — Replace Blank Strings with True Missing Values
All whitespace-only strings across categorical columns are replaced with `NaN` using regex, ensuring consistent handling of missing data downstream.

#### Step 2 — Fix Row Inconsistencies
Boolean flag columns derived from companion numeric/string columns are reconciled against their source of truth:
- `Header_Exists` ← `Header_Size_Bytes > 0`
- `Background_Exists` ← `Background_Size_Bytes > 0`
- `Has_Support_Link` ← `SupportURL != "Unknown"`
- `IsFree` ← `PriceFinal (or PriceInitial) == 0`

This prevents contradictory signals from polluting model training.

#### Step 3 — Release Date Parsing
Raw `ReleaseDate` strings are highly irregular (e.g., *"Q2 2021"*, *"coming soon"*, *"Summer 2019"*, *"1st Oct 2020"*). A custom parser handles:
- Quarter patterns (Q1–Q4 with year)
- Season mappings (Spring → March, Summer → June, etc.)
- Half-year markers ("first half", "second half")
- Ordinal suffix removal (1st → 1, 2nd → 2)
- Ambiguous/vague labels (TBD, TBA, "coming soon") → `None`

Three features are extracted and the raw column is dropped:
- `Release_Year`
- `Release_Month`
- `ReleaseDate_missing` (binary flag)

> **Milestone 1 Addition:** Cyclical month encoding (`month_sin`, `month_cos`) and a `game_age_years` feature (2024 − Release_Year) are added to preserve temporal continuity and capture time-to-accumulate-recommendations effects.

#### Step 4 — Unified Game Description
Three overlapping text columns (`DetailedDescrip`, `AboutText`, `ShortDescrip`) are collapsed into a single `Game_Description` column via a priority-based fillna chain.

#### Step 5 — Sparse Text Column Handling
Columns with very low fill rates (`SupportEmail`, `DRMNotice`, `LegalNotice`, `ExtUserAcctNotice`, platform requirement texts) are converted to binary presence flags (`Has_SupportEmail`, `Has_DRMNotice`, etc.) rather than imputed, since their actual text content carries no reliable signal.

#### Step 6 — List-Like Column Cleaning
`SupportedLanguages` and `PopularUserTags` contain semi-structured multi-value strings. Each is parsed by splitting on delimiters (`; , | / *`), stripping whitespace, deduplicating, and sorting into canonical tuples for downstream processing.

#### Step 7 — Text Column Cleaning
Trademark symbols (`™`, `®`, `©`, `℠`), parenthetical markers `(TM)`, `(R)`, and excess whitespace are stripped from text columns such as `ResponseName`, `LegalNotice`, and `Game_Description`.

#### Step 8 — Sentiment Analysis on Reviews
The `Reviews` column contains free-text user reviews. A fine-tuned DistilBERT model (`ericsonwillians/distilbert-base-uncased-steam-sentiment`) classifies each review as positive (1) or negative (0). Rows without reviews receive a sentinel value of −1.

```python
review_pipe = pipeline(
    task="sentiment-analysis",
    model="ericsonwillians/distilbert-base-uncased-steam-sentiment",
    ...
)
# Result stored in df["Review_Sentiment"]: 1 (positive), 0 (negative), -1 (no review)
```

#### Step 9 — Drop Identifier and Redundant Columns
Raw source columns that have been transformed or encoded are removed: `QueryID`, `ResponseID`, `QueryName`, `ReleaseYear`, `SupportURL`, `Background`, `HeaderImage`, `Website`, `AboutText`, `DetailedDescrip`, `ShortDescrip`, `Reviews`, and the intermediate date parsing columns.

#### Step 10 — Drop Constant Columns
Any column with ≤ 1 unique value is dropped, as it carries zero information.

#### Step 11 — Remove Duplicate Rows
Exact duplicate rows are identified and dropped, resetting the index.

#### Step 12 — Iterative Imputation (Numeric Columns)
Missing values in numeric columns are filled using `IterativeImputer`, which models each feature as a function of all others. Binary flags and helper columns (`Has_*`, `ReleaseDate_missing`, boolean columns) are excluded from imputation to preserve their meaning.

- **Milestone 1** uses a `BayesianRidge` estimator and fits only on training rows to prevent data leakage.
- **Milestone 2** uses the default estimator with `random_state=42`.

After imputation, `Release_Month` is clipped to [1, 12] and both date parts are cast to `Int64`.

#### Step 13 — Categorical Imputation
Remaining missing values in categorical columns are filled with the column mode. List-like columns retain empty tuples for missing entries.

#### Step 14 — Outlier Detection & Log Transformation
Box plots are generated for all numeric columns. Columns with skewness > 1 and all-non-negative values are log-transformed via `np.log1p` to compress right-skewed distributions. Binary flags and date columns are excluded.

> **Milestone 1 Addition:** After log transform, remaining high-skew columns are Winsorized using IQR capping (1.5× IQR bounds) to further reduce the influence of extreme outliers.

#### Step 15 — Boolean-to-Integer Conversion
All remaining `bool` dtype columns are cast to `int` (0/1) for compatibility with scikit-learn estimators.

---

### Phase 2 — Milestone 2 Additions

Milestone 2 adds the following steps on top of the shared pipeline before feature selection:

**Language & Tag Encoding:**
- `SupportedLanguages` → `language_count` (number of supported languages), `has_english` (binary)
- `PopularUserTags` → `tag_count` + 25 binary columns for the top 25 most frequent tags (`tag_action`, `tag_indie`, etc.)

**System Requirements Extraction:**
| Source Column | Extracted Features |
|---|---|
| `PCMinReqsText` | `pc_min_ram_gb`, `pc_min_word_count`, `pc_min_has_directx` |
| `PCRecReqsText` | `has_pc_rec_reqs`, `pc_rec_ram_gb`, `pc_rec_word_count` |
| `LinuxMinReqsText` | `has_linux_min_reqs` (binary) |
| `MacMinReqsText` | `has_mac_min_reqs` (binary) |

**Semantic Description Embeddings:**
Game descriptions are encoded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings), then compressed to 30 PCA components (fitted on training rows only), capturing the latent semantic meaning of game descriptions as dense numeric features (`desc_semantic_0` … `desc_semantic_29`).

**Ordinal Encoding:**
- `Header_Size_Group` is ordinally encoded in the order: `Small → Medium → Large → Extra_Large`
- `GamePopularity` target is mapped: `Low → 0`, `Medium → 1`, `High → 2`

---

## 🔧 Feature Engineering

| Feature | Source | Description |
|---|---|---|
| `Release_Year`, `Release_Month` | `ReleaseDate` | Parsed date components |
| `ReleaseDate_missing` | `ReleaseDate` | Binary flag for unparseable dates |
| `month_sin`, `month_cos` | `Release_Month` | Cyclical month encoding (Milestone 1) |
| `game_age_years` | `Release_Year` | Years since release (Milestone 1) |
| `Review_Sentiment` | `Reviews` | DistilBERT sentiment: 1/0/−1 |
| `Game_Description` | 3 text columns | Priority-merged description |
| `desc_semantic_0…29` | `Game_Description` | 30-dim PCA of MiniLM embeddings |
| `Has_SupportEmail`, etc. | Sparse text cols | Binary presence flags |
| `language_count`, `has_english` | `SupportedLanguages` | Language features |
| `tag_count`, `tag_*` (25 cols) | `PopularUserTags` | Tag count + top-25 tag dummies |
| `pc_min_ram_gb`, etc. | PC requirement text | Extracted numeric specs |

---

## 🎯 Feature Selection

### Regression Feature Selection

A single-step **Random Forest importance** method is used. All features are ranked by their contribution to the forest's ensemble predictions, and the minimal set of features whose cumulative importance reaches 95% is retained. This balances model compactness with predictive coverage.

```
Selected features covering 95% cumulative importance → used for all regression models
```

Binary features (0/1 only) are excluded from `StandardScaler` to preserve their semantics; only continuous numeric features are scaled.

---

### Classification Feature Selection

A three-step sequential pipeline progressively narrows the feature set:

**Step 1 — Variance Threshold**
Near-zero variance features (essentially constant) are removed. They contain no discriminative signal for any class.

**Step 2 — ANOVA F-Test**
`f_classif` tests whether the mean value of each feature differs significantly across the three popularity classes. Only features with `p-value < 0.05` (statistically significant) are retained.

**Step 3 — Mutual Information Classification**
`mutual_info_classif` measures how much each feature reduces uncertainty about the target class. Features with MI score above the median MI threshold are kept.

**Step 4 — Permutation Importance (Random Forest)**
A quick Random Forest is fitted on the MI-filtered set. Permutation importance identifies features whose removal (shuffling) does not hurt accuracy (mean importance ≤ 0). These are dropped as they add noise without benefit.

> The final selected features are split into two sets:
> - **Scaled features** (for distance-based models: SVM, KNN) — standardised with `StandardScaler`
> - **Tree features** (for ensemble models: XGBoost, Random Forest, Gradient Boosting) — used as-is

---

## 📈 Regression Phase (Milestone 1)

**Target:** `RecommendationCount` — continuous integer.  
**Split:** 80/20 train/test, stratified by index with `random_state=42`.  
**Validation:** 5-fold `KFold` cross-validation via `GridSearchCV` (scoring: `neg_mean_squared_error`).

### Models & Hyperparameter Tuning

| Model | Key Hyperparameters Tuned |
|---|---|
| **Linear Regression** | Baseline — no hyperparameters |
| **Ridge Regression** | `alpha` ∈ {0.1, 1.0, 10.0, 100.0} |
| **Lasso Regression** | `alpha` ∈ {0.1, 1.0, 10.0, 100.0} |
| **Decision Tree** | `max_depth` ∈ {3, 5, 10}, `min_samples_leaf` ∈ {5, 10, 20} |
| **Random Forest** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, 10}, `min_samples_leaf` ∈ {5, 10} |
| **Gradient Boosting** | `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {3, 4, 5}, `learning_rate` ∈ {0.05, 0.1, 0.2}, `min_samples_leaf` ∈ {5, 10} |
| **Extra Trees** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, 10}, `min_samples_leaf` ∈ {5, 10} |
| **XGBoost** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `tree_method='hist'` |

### Regression Results Comparison

> **Primary metric: Test R²** (higher is better). MSE is secondary (lower is better).

| Model | Train MSE | Test MSE | Train R² | Test R² |
|---|---|---|---|---|
| Linear Regression | — | — | — | Low |
| Ridge Regression | — | — | — | Low |
| Lasso Regression | — | — | — | Low |
| Decision Tree | — | — | High (overfit) | Moderate |
| **Random Forest** | Low | **Lowest** | High | **Best** |
| **Gradient Boosting** | Low | Low | High | **Best** |
| Extra Trees | Low | Low | High | High |
| XGBoost | Low | Low | High | High |

> Tree-based ensemble models (Random Forest, Gradient Boosting, Extra Trees, XGBoost) substantially outperform linear models on this dataset, as the relationship between game features and recommendation counts is highly non-linear. Linear Regression, Ridge, and Lasso produce poor Test R² scores, indicating the features cannot linearly predict recommendations. Ensemble methods capture complex feature interactions and are therefore the recommended models for this task.

**Best model by Test R²:** Gradient Boosting or Random Forest (lowest test MSE, highest generalisation R²).

---

## 🏷️ Classification Phase (Milestone 2)

**Target:** `GamePopularity` — ordinal: `Low (0)`, `Medium (1)`, `High (2)`.  
**Split:** Stratified 80/20 train/validation split.  
**Primary metric:** Macro F1-score (treats all three classes equally, robust to imbalance).

### Target Variable

```
GamePopularity mapping:
  'Low'    → 0
  'Medium' → 1
  'High'   → 2
```

### Class Imbalance — SMOTE

The training set is class-imbalanced. SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training split only to produce balanced class distributions:

```python
smote = SMOTE(random_state=42, k_neighbors=5)

# For distance-based models (SVM, KNN) — use scaled features
X_train_scaled_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

# For tree-based models (XGBoost, RF, GB) — use unscaled features
X_train_trees_sm, y_train_trees_sm = smote.fit_resample(X_train_trees, y_train)
```

> SMOTE is applied **only to training data** to prevent data leakage into validation/test sets. Additionally, `class_weight='balanced'` is set on SVM as a safety net.

### Models & Hyperparameter Tuning

#### Model 1 — Support Vector Machine (SVM)

Due to SVM's O(n²–n³) complexity, a stratified subsample of 6,000 training samples is used for experiments. `LinearSVC` is used for the linear kernel (significantly faster). Uses scaled + SMOTE-balanced features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `kernel` | linear, rbf, poly, sigmoid |
| B | `C` (regularisation) | 0.01, 0.1, 1.0, 10.0, 100.0 |
| C | `gamma` (RBF bandwidth) | scale, auto, 0.001, 0.01, 0.1 |
| D | `degree` (poly kernel) | 2, 3, 4, 5 |

**Best SVM config:** `LinearSVC`, `loss='hinge'`, `C=1.0`, `class_weight='balanced'`

---

#### Model 2 — XGBoost Classifier

Uses unscaled + SMOTE-balanced tree features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `n_estimators` | 100, 200, 300, 500 |
| B | `max_depth` | 3, 4, 6, 8 |
| C | `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| D | `colsample_bytree` | 0.6, 0.8, 1.0 |
| E | `reg_alpha` | 0, 0.1, 1 |
| F | `reg_lambda` | 0.5, 1, 2 |

**Best XGBoost config:** `colsample_bytree=1.0`, `max_depth=4`, `learning_rate=0.05`, `n_estimators=100`

---

#### Model 3 — Random Forest Classifier

Uses unscaled + SMOTE-balanced tree features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `n_estimators` | 50, 100, 200 |
| B | `max_depth` | None, 10, 20 |
| C | `min_samples_split` | 2, 10, 25 |
| D | `min_samples_leaf` | 1, 2, 5 |
| E | `max_features` | sqrt, log2 |
| F | `criterion` | gini, entropy |

**Best RF config:** `max_features='log2'`, `min_samples_split=25`, `min_samples_leaf=2`, `criterion='entropy'`, `class_weight='balanced_subsample'`, `n_estimators=200`

---

#### Model 4 — Gradient Boosting Classifier

Uses unscaled + SMOTE-balanced tree features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `n_estimators` | 100, 300, 500, 700, 900 |
| B | `max_depth` | 3, 5, 8 |
| C | `learning_rate` | 0.05, 0.1, 0.2 |
| D | `subsample` | 0.8, 0.9, 1.0 |

**Best GB config:** `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.8`

---

#### Model 5 — K-Nearest Neighbors (KNN)

Uses scaled + SMOTE-balanced features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `n_neighbors` | 3, 5, 7, 11, 15 |
| B | `metric` | euclidean, manhattan, chebyshev, minkowski |
| C | `weights` | uniform, distance |
| D | `p` (Minkowski power) | 1, 2, 3, 4 |
| E | `leaf_size` | 10, 20, 30, 50, 100 |

**Best KNN config:** `n_neighbors=3`, `metric='manhattan'`, `weights='distance'`, `leaf_size=30`

---

### Classification Results Comparison

> **Primary metric: Macro F1** (reported as the best across all hyperparameter experiments per model).

| Model | Max Accuracy | Train Time (total, s) | Inference Time (total, s) | # Experiments |
|---|---|---|---|---|
| **SVM** (LinearSVC) | Best among linear | Fast | Fastest | ~10 |
| **XGBoost** | High | Moderate | Very fast | ~18 |
| **Random Forest** | High | Moderate | Very fast | ~18 |
| **Gradient Boosting** | High | Slowest | Fast | ~11 |
| **KNN** | Lowest | Fast (fit) | Slow (predict) | ~17 |

**Key Takeaways:**
- **Most accurate:** Random Forest or XGBoost consistently achieve the highest macro F1 and accuracy across experiments.
- **Fastest training:** SVM (LinearSVC on subsampled data) and KNN.
- **Fastest inference:** SVM (LinearSVC) and XGBoost.
- **KNN** lags in accuracy due to the high dimensionality of the feature space and the difficulty of meaningful distance computation across heterogeneous features.
- **Gradient Boosting** achieves competitive accuracy but has the longest cumulative training time due to its sequential boosting nature.
- All models are saved as `.pkl` files via the `exam_bundle.pkl` artifact for reproducibility.

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| ML Models | `scikit-learn`, `xgboost` |
| NLP / Embeddings | `transformers` (HuggingFace), `sentence-transformers` |
| Deep Learning Backend | `torch` (PyTorch) |
| Imbalanced Learning | `imbalanced-learn` (SMOTE) |
| Environment | Kaggle / Python 3.10+ |

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Asma-2005/Game-Prediction-ML.git
cd Game-Prediction-ML

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn \
            torch transformers sentence-transformers

# 3. Run Milestone 1 — Regression
jupyter notebook Regression_Models.ipynb

# 4. Run Milestone 2 — Classification
jupyter notebook Classification_Models_Milestone2.ipynb
```

> **Note:** The notebooks were developed on Kaggle with GPU acceleration. Sentiment analysis and semantic embedding steps will run significantly faster with a CUDA-compatible GPU.

---
