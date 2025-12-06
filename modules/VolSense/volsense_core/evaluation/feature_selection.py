import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV


def compute_feature_correlations(df: pd.DataFrame, target_col: str = "realized_vol_log", threshold: float = 0.95):
    feature_cols = [col for col in df.columns if col not in ["date", "ticker", target_col]]
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected = [f for f in feature_cols if f not in to_drop]
    return selected, corr_matrix, to_drop


def compute_mutual_information(df: pd.DataFrame, target_col: str = "realized_vol_log"):
    X = df.drop(columns=["date", "ticker", target_col])
    y = df[target_col]
    mi = mutual_info_regression(X, y, discrete_features=False, random_state=42)
    scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return scores


def perform_recursive_feature_elimination(df: pd.DataFrame, target_col: str = "realized_vol_log", top_n: int = 20):
    X = df.drop(columns=["date", "ticker", target_col])
    y = df[target_col]
    model = RidgeCV(alphas=np.logspace(-3, 3, 7))
    selector = RFE(model, n_features_to_select=top_n)
    selector = selector.fit(X, y)
    selected_features = list(X.columns[selector.support_])
    return selected_features


def model_feature_importance(df: pd.DataFrame, target_col: str = "realized_vol_log", top_n: int = 20):
    X = df.drop(columns=["date", "ticker", target_col])
    y = df[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(top_n)


def plot_feature_heatmap(corr_matrix, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def rank_features(df: pd.DataFrame, target_col: str = "realized_vol_log", corr_thresh: float = 0.95, top_n: int = 20):
    reduced_feats, corr_matrix, dropped = compute_feature_correlations(df, target_col, threshold=corr_thresh)
    mi_scores = compute_mutual_information(df[reduced_feats + [target_col, "date", "ticker"]], target_col)
    rfe_feats = perform_recursive_feature_elimination(df[reduced_feats + [target_col, "date", "ticker"]], target_col, top_n)
    rf_scores = model_feature_importance(df[reduced_feats + [target_col, "date", "ticker"]], target_col, top_n)

    return {
        "mutual_info": mi_scores.head(top_n),
        "recursive_elimination": rfe_feats,
        "model_importance": rf_scores,
        "dropped_correlated": dropped,
        "correlation_matrix": corr_matrix
    }