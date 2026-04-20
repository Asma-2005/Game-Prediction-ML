# 🎮 Predicting Online Game Popularity
### Steam Game Recommendation Count Forecasting — Machine Learning Pipeline
 
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
</p>
 
## 📌 Project Overview
 
Can you predict how popular an online game will be? This project analyzes a dataset of **11,357 Steam games** to identify the key drivers of game popularity and build machine learning models that estimate a game's `RecommendationCount` — Steam's primary measure of player approval.
 
**Target variable:** `RecommendationCount` (number of player recommendations on Steam)
 
**Challenge:** Extremely skewed target (skewness = 68.0), 63% zero-inflation, and a mix of numeric, boolean, text, and date features requiring careful preprocessing.
 
---
 
## 📊 Dataset Description
 
| Property | Value |
|---|---|
| Training samples | 11,357 games |
| Total features | 78 columns |
| Feature types | Numeric (18) · Boolean (30) · Text (18) · Date/ID/URL (12) |
| Target | `RecommendationCount` (continuous, heavily skewed) |
| Platform | Steam (Valve) |
| Missing values | Only `Website` (23.9%) — minimal overall |
 
### Key Feature Groups
 
| Group | Examples |
|---|---|
| **Game metadata** | `ReleaseDate`, `RequiredAge`, `Metacritic`, `DLCCount` |
| **Platform flags** | `PlatformWindows`, `PlatformLinux`, `PlatformMac` |
| **Category flags** | `CategoryMultiplayer`, `CategoryCoop`, `CategoryMMO` |
| **Genre flags** | `GenreIsAction`, `GenreIsIndie`, `GenreIsFreeToPlay` |
| **SteamSpy data** | `SteamSpyOwners`, `SteamSpyPlayersEstimate` ⚠️ potential leakage |
| **Price info** | `PriceInitial`, `PriceFinal`, `PriceCurrency` |
| **Text columns** | `AboutText`, `DetailedDescrip`, `ShortDescrip`, `Reviews` |
| **Media counts** | `MovieCount`, `ScreenshotCount`, `AchievementCount` |
 
---
 
## 🔍 Key EDA Findings
 
### Target Distribution
 
```
Mean:     1,214      ← inflated by extreme outliers
Median:   0          ← 63.4% of games have zero recommendations
Skewness: 68.0       ← extremely right-skewed (normal ≈ 0)
Max:      1,427,633  ← CS:GO-tier games
P90:      1,253      ← top 10% starts here
P99:      20,543     ← top 1% starts here
```
 
> **Decision:** Apply `log1p()` transform to the target before training. Invert with `expm1()` after prediction.
