# 🍷 Wine Project: Predicting Red Wine Ratings

## Project Structure

```
├── config.yaml                    # Project configuration settings
├── README.md                      # Project documentation

├── data/
│   └── raw/
│       ├── red.csv                # Original Vivino red wine dataset
│       ├── varieties.csv          # Reference list of grape varieties
│       ├── red_ml_final.csv       # Cleaned dataset with engineered features
│       └── red_ml_model_input.csv # Minimal dataset for modelling

├── notebooks/
│   ├── data_cleaning_beatriz_update.ipynb   # Data wrangling & variety engineering
│   ├── data_modelling_victoria.ipynb        # EDA, feature engineering & KNN model
│   └── insights-storytelling_rachel.ipynb   # Feature importance, value analysis & storytelling
```

---

## Project Overview

Can an algorithm predict how good a wine tastes? That was the core question behind this project.

Using the [Vivino Red Wine dataset from Kaggle](https://www.kaggle.com/datasets/budnyak/wine-rating-and-price), we built a machine learning model to predict red wine ratings based on features like price, country, grape variety, and wine age. We limited our scope to red wines only to keep the business question focused: **what actually drives a high Vivino rating?**

Beyond the model, we explored patterns in the data using Tableau — looking at how country, winery, and variety relate to perceived quality and value.

---

## Business Questions

- What makes a red wine highly rated on Vivino?
- Does price strongly correlate with rating?
- Which regions consistently offer better value?
- Which features most influence wine quality?
- Where do mismatches happen — expensive but low-rated wines?

---

## Dataset

**Source:** [Kaggle – Wine Rating and Price](https://www.kaggle.com/datasets/budnyak/wine-rating-and-price)

We used only `red.csv` to keep the scope clear and the model focused. `varieties.csv` was used as a reference for grape name matching but not merged directly, as it lacked reliable join keys.

| File | Description |
|---|---|
| `red.csv` | Main dataset: name, winery, country, region, price, rating, year, number of ratings |
| `varieties.csv` | Reference list of known grape variety names |


---

## Data Cleaning & Feature Engineering

Raw data required significant preprocessing before it could be used for modelling:

1. **Column standardisation** – Column names were lowercased and stripped of whitespace for consistency across notebooks.
2. **Text normalisation** – Wine names and regions were normalised using Unicode NFKD encoding and regex to remove accents and special characters.
3. **Year cleaning** – The `year` column was coerced to numeric, and invalid or missing values were dropped. A derived `wine_age` feature was created (2026 − year).
4. **Log transformation** – `price`, `numberofratings`, and `wine_age` were right-skewed and were log-transformed and standardised to better suit the model.

### Engineering `variety_ml`

This was one of the most challenging parts of the project. The dataset contained no explicit grape variety column — only wine names and regions, which are often vague or producer-branded. We extracted variety information through:

- **Regex pattern matching** against a known variety list (longest match first to avoid partial clashes)
- **Region-based inference** using a hand-curated mapping (e.g., `pomerol` → Bordeaux Blend, `barolo` → Nebbiolo, `rioja` → Tempranillo)
- **Fallback categories** for wines that couldn't be identified:
  - `Rare Varieties` – real but uncommon grapes too sparse to keep as individual classes (e.g., Tannat, Mourvèdre, Gamay)
  - `Unspecified Red` – wines with intentionally vague labels (e.g., "Rosso Toscana", "Red Wine")
  - `Unknown Variety` – no identifiable grape or region information (e.g., "Grande Cuvée", "Tradition")

---
### Feature Importance (Random Forest)

A Random Forest Regressor was trained to rank which features drive high ratings. The top predictors were dominated by **price** and **specific country/variety combinations**, confirming that origin and market positioning matter more than raw numerical features alone.

---

## Exploratory Data Analysis

### Correlation Analysis

A Pearson correlation and pairplot confirmed that:

- **Price** has the strongest correlation with rating among numerical features
- **Year / wine age** contributes some predictive power
- **NumberOfRatings** adds very little signal
- Features show low inter-correlation — minimal multicollinearity risk

## Modelling

**Goal:** Predict red wine ratings based on features  
**Model:** KNeighborsRegressor  
**k-value:** 45  
**Preprocessing:** MinMaxScaler and Log Transformation for Price, Number of Ratings, Wine Age  
**Target:** Rating  
**Features:** Price, Wine Age, Number of Ratings, Winery  
**R² score:** 0.58  
**Model accuracy:** 57.87%

In this project, we developed a K-Nearest Neighbours (KNN) regression model to predict wine ratings based on several features derived from the dataset.

First, exploratory data analysis was conducted to understand relationships between variables and detect skewed distributions. Since some numeric variables were strongly right-skewed, log transformations were applied to price, number of ratings and wine age to improve their distributions. Additionally, feature scaling (MinMax scaling) was applied to variables to ensure that all numerical features were on comparable scales, which is important for distance-based algorithms like KNN.

We then performed feature engineering, including the creation of a wine age variable derived from the wine's production year. The dataset was split into training and testing sets using an 80/20 split to evaluate the model's ability to generalize to unseen data.

Categorical variables such as country and winery were converted into numerical format using one-hot encoding, allowing them to be used as input features in the model.

The KNN regression model was trained on the prepared training dataset. Since KNN is a distance-based algorithm, feature scaling was necessary to prevent variables with larger ranges from dominating the distance calculations. Different numbers of neighbors (k values) were tested to determine the model configuration that produced the best predictive performance. Ultimately the best k-value was 45. After testing a few combinations for features the best result was given by the combination of the numeric values and winery.

Finally, the trained model was evaluated on the test dataset using the R² score, which measures how well the predicted ratings explain the variation in the actual wine ratings.

Although we recognize that other models such as Random Forest could potentially perform better for predicting ratings using high-cardinality categorical variables like winery and country, the KNN approach still provided valuable insights. In particular, the analysis highlighted the strong influence of price and number of ratings on wine ratings, helping us better understand the key factors associated with higher-rated wines.

---

## Key Insights

**Insight 1 – Price is the strongest predictor, but not the whole story.**  
Price topped the feature importance chart, likely acting as a proxy for brand reputation and regional prestige rather than a direct driver of quality. Scatter plots confirm that at any given price point, ratings remain tightly clustered — high price does not guarantee a high rating.

**Insight 2 – Origin matters.**  
Certain countries consistently produce higher-rated wines. Country was one of the more useful categorical features, adding meaningful signal to the model beyond what price alone provides.

**Insight 3 – Value exists outside premium regions.**  
Some less traditional wine-producing countries (filtered to those with more than 20 wines) offer high ratings relative to price. The value score (rating ÷ price) highlights where consumers can find quality without paying a premium.

---

## Setbacks & Honest Reflections

**KNN has limits for this problem.** KNN struggles with high-cardinality categorical features after one-hot encoding — adding all categorical variables (winery, region, variety, name) actually reduced R², suggesting they introduced more noise than signal. Tree-based models like Random Forest or Gradient Boosting would likely handle this better, and we explored feature importance with Random Forest for storytelling purposes.

**`variety_ml` was expensive to build.** Extracting grape variety from unstructured wine names required a combination of regex, region inference, and manual rule-curation. It was one of the most time-consuming parts of the project — and it still doesn't cover every case perfectly. That said, we're proud of how much structure we managed to extract from what was essentially free text.

**Rating prediction is inherently hard.** Wine ratings are subjective, reviewer-influenced, and contextual. Even the best-structured model will have a ceiling on how much variance it can explain from price, region, and variety alone.

---

## Visualisations

Visual exploration was performed using Python and Tableau.

**Python libraries used:**

- `pandas` – data manipulation and feature engineering
- `numpy` – numerical operations and log transformations
- `matplotlib` & `seaborn` – histograms, scatterplots, correlation heatmaps, bar charts
- `scipy` – KS normality tests and Q-Q plots
- `sklearn` – modelling, scaling, train/test split, KNN, Random Forest

**Tableau** was used for interactive exploration of country, winery, and variety-level patterns, allowing drill-down by region and price tier.
https://public.tableau.com/app/profile/rachel.vianna/viz/Vivino_Insights/Dashboard1

**Prezi Presentation**
https://prezi.com/p/cazr8sktltmc/?present=1

---

## Libraries

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
pyyaml
```

---

## Authors

Beatriz Fernandes · Rachel Vianna · Victoria Cano

*Bootcamp group project – Week 7*
