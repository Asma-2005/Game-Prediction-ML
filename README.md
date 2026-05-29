<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-green?logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<h1 align="center">🎮 Steam Game Success Forecasting</h1>

<p align="center">
  An end-to-end machine learning project focused on forecasting Steam game success using regression, classification, NLP, sentiment analysis, and ensemble learning techniques.
</p>

<p align="center">
  The project predicts both <b>game recommendation counts</b> and <b>popularity tiers</b> (Low / Medium / High) through a two-phase predictive analytics pipeline built on real Steam platform data.
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Engineering Summary](#-feature-engineering-summary)
- [Feature Selection](#-feature-selection)
  - [Milestone 1 — Regression Feature Selection](#milestone-1--regression-feature-selection)
  - [Milestone 2 — Classification Feature Selection](#milestone-2--classification-feature-selection)
- [Regression Phase — Milestone 1](#-regression-phase--milestone-1)
  - [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
  - [Regression Results Comparison](#regression-results-comparison)
- [Classification Phase — Milestone 2](#-classification-phase--milestone-2)
  - [Target Variable](#target-variable)
  - [Class Imbalance — SMOTE](#class-imbalance--smote)
  - [Models & Hyperparameter Tuning](#models--hyperparameter-tuning-1)
  - [Classification Results Comparison](#classification-results-comparison)
- [Milestone Comparison at a Glance](#-milestone-comparison-at-a-glance)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)

---

## 🔍 Project Overview

This project applies a full machine learning pipeline to a Steam games dataset to answer two prediction problems:

| Phase | Task | Target | Notebook |
|---|---|---|---|
| Milestone 1 | **Regression** | `RecommendationCount` (continuous) | `Regression_Models.ipynb` |
| Milestone 2 | **Classification** | `GamePopularity` (Low / Medium / High) | `Classification_Models_Milestone2.ipynb` |

Both milestones share an identical preprocessing pipeline. **The only difference between them is the feature selection strategy** — Milestone 1 uses Random Forest cumulative importance, while Milestone 2 uses a three-step statistical funnel (ANOVA → Mutual Information → Permutation Importance).

---

## 📁 Repository Structure

```
Game-Prediction-ML/
│
├── Data/
│   ├── Data_Preprocessing.ipynb            # Milestone 1 preprocessing notebook
│   ├── Data_Preprocessing_milestone2.ipynb # Milestone 2 preprocessing (feature selection only differs)
│   ├── EDA.ipynb                           # Exploratory data analysis
│   ├── data_Extended.ipynb                 # Extended data collection
│   ├── data_preprocessed.csv               # Cleaned & preprocessed dataset (Milestone 1)
│   ├── data_preprocessed_MS2.csv           # Cleaned & preprocessed dataset (Milestone 2)
│   ├── data_urls_final.csv                 # Raw dataset with URLs
│   ├── dataset_API.csv                     # API-fetched raw data
│   ├── idlist.csv                          # Game ID list used for API fetching
│   ├── train_data.csv                      # Training split with RecommendationCount column (Milestone 1)
│   └── train_data_2.csv                    # Training split with GamePopularity column (Milestone 2)
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
- **Classification:** `GamePopularity` — an ordinal label (`Low`, `Medium`, `High`) sourced from `train_data_2.csv` and joined to the main dataset.

---

## 🧹 Data Preprocessing

The preprocessing pipeline below is **fully shared between both milestones**. Milestone 2 does not add any new preprocessing steps — it only differs in how features are selected after this pipeline completes (see [Feature Selection](#-feature-selection)).

---

#### Step 1 — Replace Blank Strings with True Missing Values
All whitespace-only strings across categorical columns are replaced with `NaN` using regex, ensuring consistent missing-value handling downstream.

---

#### Step 2 — Fix Row Inconsistencies
Boolean flag columns derived from companion numeric/string columns are reconciled against their source of truth:
- `Header_Exists` ← `Header_Size_Bytes > 0`
- `Background_Exists` ← `Background_Size_Bytes > 0`
- `Has_Support_Link` ← `SupportURL != "Unknown"`
- `IsFree` ← `PriceFinal (or PriceInitial) == 0`

This prevents contradictory signals from polluting model training.

---

#### Step 3 — Release Date Parsing
Raw `ReleaseDate` strings are highly irregular (e.g., *"Q2 2021"*, *"coming soon"*, *"Summer 2019"*, *"1st Oct 2020"*). A custom parser handles:
- Quarter patterns (Q1–Q4 with year)
- Season mappings (Spring → March, Summer → June, etc.)
- Half-year markers ("first half", "second half")
- Ordinal suffix removal (1st → 1, 2nd → 2)
- Ambiguous/vague labels (TBD, TBA, "coming soon") → `None`

Three features are extracted and the raw column is dropped:
- `Release_Year`, `Release_Month`, `ReleaseDate_missing` (binary flag)

> **Regression addition:** Cyclical month encoding (`month_sin`, `month_cos`) and a `game_age_years` feature (`2024 − Release_Year`) are added to preserve temporal continuity and capture time-to-accumulate-recommendations effects.

---

#### Step 4 — Unified Game Description
Three overlapping text columns (`DetailedDescrip`, `AboutText`, `ShortDescrip`) are collapsed into a single `Game_Description` column via a priority-based fillna chain.

---

#### Step 5 — Sparse Text Column Handling
Columns with very low fill rates (`SupportEmail`, `DRMNotice`, `LegalNotice`, `ExtUserAcctNotice`, platform requirement texts) are converted to binary presence flags (`Has_SupportEmail`, `Has_DRMNotice`, etc.) rather than imputed, since their actual text content carries no reliable signal.

---

#### Step 6 — List-Like Column Cleaning
`SupportedLanguages` and `PopularUserTags` contain semi-structured multi-value strings. Each is parsed by splitting on delimiters (`; , | / *`), stripping whitespace, deduplicating, and sorting into canonical tuples for downstream processing.

---

#### Step 7 — Text Column Cleaning
Trademark symbols (`™`, `®`, `©`, `℠`), parenthetical markers `(TM)`, `(R)`, and excess whitespace are stripped from text columns such as `ResponseName`, `LegalNotice`, and `Game_Description`.

---

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

---

#### Step 9 — Drop Identifier and Redundant Columns
Raw source columns that have already been transformed or encoded are removed: `QueryID`, `ResponseID`, `QueryName`, `ReleaseYear`, `SupportURL`, `Background`, `HeaderImage`, `Website`, `AboutText`, `DetailedDescrip`, `ShortDescrip`, `Reviews`, and the intermediate date-parsing columns.

---

#### Step 10 — Drop Constant Columns
Any column with ≤ 1 unique value is dropped, as it carries zero information.

---

#### Step 11 — Remove Duplicate Rows
Exact duplicate rows are identified and dropped, resetting the index.

---

#### Step 12 — Train / Test Split
An 80/20 stratified split is performed **before** fitting any transformer (imputer, scaler, PCA) so that all subsequent fit steps are applied only to training rows, preventing data leakage.

---

#### Step 13 — Iterative Imputation (Numeric Columns)
Missing values in numeric columns are filled using `IterativeImputer` (BayesianRidge estimator), which models each feature as a function of all others. Binary flags and helper columns (`Has_*`, `ReleaseDate_missing`, boolean columns) are excluded from imputation to preserve their semantics. After imputation, `Release_Month` is clipped to [1, 12] and cast to `Int64`.

---

#### Step 14 — Categorical Imputation
Remaining missing values in categorical columns are filled with the column mode. List-like columns retain empty tuples for missing entries.

---

#### Step 15 — Outlier Detection & Log Transformation
Box plots are generated for all numeric columns. Columns with skewness > 1 and all-non-negative values are log-transformed via `np.log1p`. After log transform, remaining high-skew columns are Winsorized using IQR capping (1.5× IQR bounds) to further reduce the influence of extreme outliers. Binary flags and date columns are excluded throughout.

---

#### Step 16 — Boolean-to-Integer Conversion
All remaining `bool` dtype columns are cast to `int` (0/1) for compatibility with scikit-learn estimators.

---

#### Step 17 — Language & Tag Encoding
- `SupportedLanguages` → `language_count` (number of supported languages) + `has_english` (binary)
- `PopularUserTags` → `tag_count` + **25 binary columns** for the top 25 most frequent tags (`tag_action`, `tag_indie`, etc.)

---

#### Step 18 — System Requirements Feature Extraction

| Source Column | Extracted Features |
|---|---|
| `PCMinReqsText` | `pc_min_ram_gb`, `pc_min_word_count`, `pc_min_has_directx` |
| `PCRecReqsText` | `has_pc_rec_reqs`, `pc_rec_ram_gb`, `pc_rec_word_count` |
| `LinuxMinReqsText` | `has_linux_min_reqs` (binary) |
| `MacMinReqsText` | `has_mac_min_reqs` (binary) |

---

#### Step 19 — Semantic Description Embeddings
Game descriptions are encoded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings), then compressed to **30 PCA components** (fitted on training rows only), producing dense numeric features `desc_semantic_0` … `desc_semantic_29` that capture the latent semantic meaning of each game's description.

---

#### Step 20 — Composite Feature Engineering & Encoding
- **Composite features:** `has_discount`, `discount_amount`, `platform_count`, `category_count`, `genre_count`, `language_count_log`
- **Ordinal encoding:** `Header_Size_Group` → `Small → Medium → Large → Extra_Large`
- **One-hot encoding:** `Support_Tier`, `Website_Authority`, and remaining object-type categorical columns

---

## 🔧 Feature Engineering Summary

| Feature | Source | Description |
|---|---|---|
| `Release_Year`, `Release_Month` | `ReleaseDate` | Parsed date components |
| `ReleaseDate_missing` | `ReleaseDate` | Binary flag for unparseable dates |
| `month_sin`, `month_cos` | `Release_Month` | Cyclical month encoding *(Milestone 1 only)* |
| `game_age_years` | `Release_Year` | Years since release *(Milestone 1 only)* |
| `Review_Sentiment` | `Reviews` | DistilBERT sentiment: 1 / 0 / −1 |
| `Game_Description` | 3 text columns | Priority-merged description |
| `desc_semantic_0…29` | `Game_Description` | 30-dim PCA of MiniLM-L6 embeddings |
| `Has_SupportEmail`, etc. | Sparse text cols | Binary presence flags |
| `language_count`, `has_english` | `SupportedLanguages` | Language features |
| `tag_count`, `tag_*` (25 cols) | `PopularUserTags` | Tag count + top-25 tag dummies |
| `pc_min_ram_gb`, etc. | PC requirement text | Extracted numeric specs |
| `platform_count`, `category_count` | List-type cols | Composite engineered features |

---

## 🎯 Feature Selection

Feature selection is the **only step that differs** between the two milestones.

### Milestone 1 — Regression Feature Selection

A single-step **Random Forest cumulative importance** method is used. All features are ranked by their contribution to the ensemble's predictions, and the minimal set whose cumulative importance reaches **95%** is retained. This balances compactness with predictive coverage.

```
All features → RF importance ranking → keep features summing to ≥ 95% importance
```

Binary features (0/1 only) are excluded from `StandardScaler`; only continuous numeric features are standardised.

---

### Milestone 2 — Classification Feature Selection

A three-step sequential funnel progressively narrows the feature set using statistical and model-based criteria:

**Step 1 — ANOVA F-Test**
`f_classif` tests whether the mean value of each feature differs significantly across the three popularity classes. Only features with `p-value < 0.05` are retained. This removes features whose distribution is statistically indistinguishable across classes.

**Step 2 — Mutual Information Classification**
`mutual_info_classif` measures how much each feature reduces uncertainty about the target class (including non-linear dependencies). Features with MI score above the median MI threshold are kept.

**Step 3 — Permutation Importance (Random Forest)**
A quick Random Forest is fitted on the MI-filtered set. Features whose mean permutation importance is ≤ 0 — meaning shuffling them does not hurt accuracy — are dropped as they add noise without benefit.

```
All features
    │
    ▼  ANOVA F-test (p < 0.05)
    │
    ▼  Mutual Information (score > median threshold)
    │
    ▼  Permutation Importance (mean > 0)
    │
    ▼  Final selected features
       ├── Scaled set   → SVM, KNN  (StandardScaler applied)
       └── Tree set     → XGBoost, Random Forest, Gradient Boosting  (no scaling)
```

---

## 📈 Regression Phase — Milestone 1

**Target:** `RecommendationCount` — continuous integer.  
**Split:** 80/20 train/test, `random_state=42`.  
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

| Model | Generalisation | Notes |
|---|---|---|
| Linear Regression | ❌ Poor | Cannot capture non-linear feature interactions |
| Ridge Regression | ❌ Poor | Regularisation helps slightly but relationship is non-linear |
| Lasso Regression | ❌ Poor | Sparse solution; many features zeroed out |
| Decision Tree | ⚠️ Moderate | Prone to overfitting; high train R², lower test R² |
| **Random Forest** | ✅ Best | Low test MSE, highest generalisation R² |
| **Gradient Boosting** | ✅ Best | Comparable to RF; best tuned configuration |
| Extra Trees | ✅ High | Competitive with RF and GB |
| XGBoost | ✅ High | Fast and accurate; very close to best |

> Tree-based ensemble models substantially outperform linear models because the relationship between game features and recommendation counts is highly non-linear. Linear Regression, Ridge, and Lasso produce poor Test R² scores, indicating they cannot adequately model the feature-target relationship. **Random Forest and Gradient Boosting are the recommended models** for this task, achieving the lowest test MSE and highest generalisation R².

---

## 🏷️ Classification Phase — Milestone 2

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

The training set is class-imbalanced. SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training split **only**, generating synthetic minority samples to balance the three classes. Two variants are produced to match the requirements of different model families:

```python
smote = SMOTE(random_state=42, k_neighbors=5)

# Distance-based models (SVM, KNN) — requires standardised features
X_train_scaled_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

# Tree-based models (XGBoost, RF, GB) — raw features, no scaling needed
X_train_trees_sm, y_train_trees_sm = smote.fit_resample(X_train_trees, y_train)
```

> SMOTE is applied **only to training data** to prevent synthetic samples from leaking into validation/test evaluation. `class_weight='balanced'` is additionally set on SVM as a safety net.

---

### Models & Hyperparameter Tuning

#### Model 1 — Support Vector Machine (SVM)

Due to SVM's O(n²–n³) complexity, a **stratified subsample of 6,000 training samples** is used across hyperparameter experiments. `LinearSVC` is used for the linear kernel (significantly faster than `SVC`). Uses **scaled + SMOTE-balanced** features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `kernel` | linear, rbf, poly, sigmoid |
| B | `C` (regularisation) | 0.01, 0.1, 1.0, 10.0, 100.0 |
| C | `gamma` (RBF bandwidth) | scale, auto, 0.001, 0.01, 0.1 |
| D | `degree` (poly kernel) | 2, 3, 4, 5 |

**Best SVM config:** `LinearSVC`, `loss='hinge'`, `C=1.0`, `class_weight='balanced'`

---

#### Model 2 — XGBoost Classifier

Uses **unscaled + SMOTE-balanced** tree features.

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

Uses **unscaled + SMOTE-balanced** tree features.

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

Uses **unscaled + SMOTE-balanced** tree features.

| Experiment | Varied Parameter | Values Tested |
|---|---|---|
| A | `n_estimators` | 100, 300, 500, 700, 900 |
| B | `max_depth` | 3, 5, 8 |
| C | `learning_rate` | 0.05, 0.1, 0.2 |
| D | `subsample` | 0.8, 0.9, 1.0 |

**Best GB config:** `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.8`

---

#### Model 5 — K-Nearest Neighbors (KNN)

Uses **scaled + SMOTE-balanced** features.

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

> **Primary metric: Macro F1** — treats all three classes equally, robust to class imbalance.

| Model | Accuracy | Train Time | Inference Time | Strengths | Weaknesses |
|---|---|---|---|---|---|
| **SVM** (LinearSVC) | Good | Fast (subsampled) | Fastest | Simple, interpretable, efficient | Limited by linear decision boundary; requires subsampling at scale |
| **XGBoost** | High | Moderate | Very fast | Strong accuracy, handles non-linearity well | Sensitive to hyperparameter tuning |
| **Random Forest** | High | Moderate | Very fast | Robust, low variance, handles mixed features | Slower than XGBoost at inference for large forests |
| **Gradient Boosting** | High | Slowest | Fast | Competitive accuracy, good generalisation | Sequential training makes it the most time-expensive |
| **KNN** | Lowest | Fast (fit only) | Slow (predict) | No training phase, conceptually simple | Struggles with high-dimensional heterogeneous features; slow prediction |

**Key Takeaways:**
- **Random Forest and XGBoost** consistently achieve the highest macro F1 and accuracy across all experiments.
- **KNN** lags in accuracy due to the curse of dimensionality — meaningful distance computation breaks down across heterogeneous, high-dimensional features.
- **Gradient Boosting** is competitive but accumulates the most training time due to its sequential nature.
- **SVM (LinearSVC)** is the fastest overall when subsampling is applied and is a strong baseline for linear separability.
- All best models are serialised via `pickle` into an `exam_bundle.pkl` for reproducible evaluation.

---

## ⚖️ Milestone Comparison at a Glance

| Aspect | Milestone 1 — Regression | Milestone 2 — Classification |
|---|---|---|
| **Target** | `RecommendationCount` (continuous) | `GamePopularity` (Low / Medium / High) |
| **Extra data source** | None | `train_data_2.csv` (GamePopularity labels) |
| **Preprocessing pipeline** | Full shared pipeline | Identical shared pipeline |
| **Release date extras** | `month_sin`, `month_cos`, `game_age_years` | Not added |
| **Feature selection** | RF cumulative importance ≥ 95% | ANOVA → Mutual Information → Permutation Importance |
| **Class imbalance** | N/A | SMOTE on training set only |
| **Scaling** | StandardScaler on non-binary numeric cols | Two variants: scaled (SVM/KNN) and unscaled (trees) |
| **Models** | 8 regressors | 5 classifiers |
| **CV strategy** | 5-fold KFold, `GridSearchCV` | Per-hyperparameter experiments + final combined run |
| **Primary metric** | Test R² and MSE | Macro F1-score |
| **Model persistence** | Not saved | Saved as `.pkl` in `exam_bundle.pkl` |

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

> **Note:** The notebooks were developed on Kaggle with GPU acceleration. The sentiment analysis (Step 8) and semantic embedding (Step 19) steps will run significantly faster with a CUDA-compatible GPU. Both steps fall back to CPU automatically if no GPU is available.
