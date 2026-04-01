# from src.insights_functions import ...
# from insights_functions import (
    #build_feature_importance_df,
    #plot_feature_importance,
    #plot_price_vs_rating,
    #add_value_score,
    #get_best_value_countries,
    #plot_best_value_countries
#)

"""
insights_functions.py

Reusable helper functions for the wine project insights/storytelling notebook.

These functions are designed to:
- clean feature names for presentation
- build and plot feature importance
- plot price vs rating
- create and analyze value score
- plot best-value countries

Expected dataframe column names:
- country
- region
- rating
- price
- variety_ml
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Global chart style
sns.set_style("whitegrid")

# Project color
WINE_RED = "#BA1628"


def clean_feature_name(name: str) -> str:
    """
    Clean one-hot encoded feature names so they look better in charts.

    Examples:
    - 'variety_ml_pinot noir' -> 'Pinot Noir'
    - 'country_france' -> 'France'
    """
    name = name.replace("variety_ml_", "")
    name = name.replace("country_", "")
    name = name.replace("_", " ")
    return name.title()


def build_feature_importance_df(
    red_model: pd.DataFrame,
    target_col: str = "rating",
    random_state: int = 42,
    n_estimators: int = 100
) -> pd.DataFrame:
    """
    Train a RandomForestRegressor and return a sorted dataframe
    of feature importances.

    Parameters
    ----------
    red_model : pd.DataFrame
        ML-ready dataframe containing predictors and target.
    target_col : str, default='rating'
        Name of the target column.
    random_state : int, default=42
        Random seed for reproducibility.
    n_estimators : int, default=100
        Number of trees in the random forest.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - feature
        - importance
        - feature_clean
    """
    X = pd.get_dummies(red_model.drop(columns=[target_col]), drop_first=True)
    y = red_model[target_col]

    model = RandomForestRegressor(
        random_state=random_state,
        n_estimators=n_estimators
    )
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importance_df["feature_clean"] = importance_df["feature"].apply(clean_feature_name)

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 5,
    color: str = WINE_RED,
    figsize: tuple = (8, 4)
) -> None:
    """
    Plot the top N most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output from build_feature_importance_df().
    top_n : int, default=5
        Number of top features to display.
    color : str, default=WINE_RED
        Bar color.
    figsize : tuple, default=(8, 4)
        Figure size.
    """
    plot_df = importance_df.head(top_n).copy()

    plt.figure(figsize=figsize)
    plt.barh(plot_df["feature_clean"], plot_df["importance"], color=color)

    plt.title("Top Drivers of Wine Ratings", fontsize=14)
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_price_vs_rating(
    df: pd.DataFrame,
    max_price: float | None = None,
    color: str = WINE_RED,
    alpha: float = 0.3,
    figsize: tuple = (6, 4)
) -> None:
    """
    Plot price vs rating, optionally filtering very expensive wines
    for cleaner storytelling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least 'price' and 'rating'.
    max_price : float or None, default=None
        If provided, only wines below this price are plotted.
    color : str, default=WINE_RED
        Point color.
    alpha : float, default=0.3
        Point transparency.
    figsize : tuple, default=(6, 4)
        Figure size.
    """
    plot_df = df.copy()

    if max_price is not None:
        plot_df = plot_df[plot_df["price"] < max_price]

    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=plot_df,
        x="price",
        y="rating",
        color=color,
        alpha=alpha
    )

    title = "Price vs Rating"
    if max_price is not None:
        title = f"Price vs Rating (Wines Under €{int(max_price)})"

    plt.title(title, fontsize=14)
    plt.xlabel("Price (€)")
    plt.ylabel("Rating")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def add_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a value_score column defined as rating / price.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'rating' and 'price'.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with added 'value_score' column.
    """
    df = df.copy()
    df["value_score"] = df["rating"] / df["price"]
    return df


def get_best_value_countries(
    df: pd.DataFrame,
    min_wines: int = 20,
    top_n: int = 10
) -> pd.Series:
    """
    Return the countries with the highest average value_score,
    after filtering out countries with too few wines.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'country', 'rating', and 'price',
        or already containing 'value_score'.
    min_wines : int, default=20
        Minimum number of wines required for a country to be included.
    top_n : int, default=10
        Number of countries to return.

    Returns
    -------
    pd.Series
        Sorted series of top countries by average value_score.
    """
    working_df = df.copy()

    if "value_score" not in working_df.columns:
        working_df = add_value_score(working_df)

    value_by_country = (
        working_df.groupby("country")
        .filter(lambda x: len(x) > min_wines)
        .groupby("country")["value_score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return value_by_country


def plot_best_value_countries(
    df: pd.DataFrame,
    min_wines: int = 20,
    top_n: int = 10,
    color: str = WINE_RED,
    figsize: tuple = (8, 4)
) -> None:
    """
    Plot the countries with the highest average value_score.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'country', 'rating', and 'price',
        or already containing 'value_score'.
    min_wines : int, default=20
        Minimum number of wines required for a country to be included.
    top_n : int, default=10
        Number of countries to plot.
    color : str, default=WINE_RED
        Bar color.
    figsize : tuple, default=(8, 4)
        Figure size.
    """
    value_by_country = get_best_value_countries(
        df,
        min_wines=min_wines,
        top_n=top_n
    )

    plt.figure(figsize=figsize)
    value_by_country.plot(kind="barh", color=color)

    plt.title("Best Value Wine Countries", fontsize=14)
    plt.xlabel("Rating per € (higher = better value)")
    plt.ylabel("")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.2)

    plt.figtext(
        0.5,
        -0.05,
        "Value = average rating divided by price. Higher values mean better quality for the price.",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

