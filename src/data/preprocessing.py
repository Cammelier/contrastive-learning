
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


def drop_nans(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def query_filter(df: pd.DataFrame, query: Optional[str] = None) -> pd.DataFrame:
    if query:
        return df.query(query)
    return df


def rare_category_filter(
    df: pd.DataFrame, cat_cols: list, min_count: int = 3000
) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < min_count].index
        df = df[~df[col].isin(rare_categories)]
    return df


def subsample_df(
    df: pd.DataFrame,
    n_samples: int,
    random_state: Optional[int] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    if label_col is None:
        if n_samples > len(df):
            return df.sample(n=len(df), random_state=random_state)
        return df.sample(n=n_samples, random_state=random_state)
    else:
        return (
            df.groupby(label_col, group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), n_samples // df[label_col].nunique()),
                    random_state=random_state,
                )
            )
            .reset_index(drop=True)
        )


def equalize_classes(
    df: pd.DataFrame,
    label_col: str,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    min_count = df[label_col].value_counts().min()
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=random_state))
        .reset_index(drop=True)
    )


def ml_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: Optional[int] = None,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1.0")

    train_df, remaining_df = train_test_split(
        df, train_size=train_frac, random_state=random_state, stratify=df[label_col]
    )

    val_size = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        remaining_df,
        train_size=val_size,
        random_state=random_state,
        stratify=remaining_df[label_col],
    )

    return train_df, val_df, test_df


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = pd.DataFrame(X).copy()
        self.lower_bounds_ = X.quantile(self.lower_quantile)
        self.upper_bounds_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


class TopNCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 32, default_value: int = 0):
        self.top_n = top_n
        self.default_value = default_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = pd.DataFrame(X).copy()
        self.category_maps_ = {}
        n_categories = self.top_n - 1  # Reserve 0 for default
        for col in X.columns:
            top_categories = X[col].value_counts().nlargest(n_categories).index.tolist()

            category_map = {
                category: idx + 1 for idx, category in enumerate(top_categories)
            }
            self.category_maps_[col] = category_map
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            category_map = self.category_maps_.get(col, {})
            X[col] = X[col].map(category_map).fillna(self.default_value).astype(int)
        return X
