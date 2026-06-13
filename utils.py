"""
utils.py
--------
Utility functions for the BSc Thesis: Comparing Traditional ML, AutoML and
Tabular Foundation Models for Bank Customer Churn Prediction.

Author: Yigit Ali Uysal
Supervisor: Elias Dubbeldam
University of Amsterdam, 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def load_and_preprocess(filepath='Churn_Modelling.csv'):
    """
    Load and preprocess the Bank Customer Churn dataset.

    Returns separate feature matrices for XGBoost (one-hot encoded)
    and AutoGluon/TabPFN (raw categorical features).

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    X_xgb : pd.DataFrame
        One-hot encoded features for XGBoost.
    X_auto : pd.DataFrame
        Raw features for AutoGluon and TabPFN.
    y : pd.Series
        Target variable (Exited).
    """
    df = pd.read_csv(filepath)
    df_clean = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    y = df_clean['Exited'].reset_index(drop=True)
    X_raw = df_clean.drop(columns=['Exited']).reset_index(drop=True)

    # One-hot encode for XGBoost
    X_xgb = pd.get_dummies(X_raw, columns=['Geography', 'Gender'], drop_first=False)

    # Raw features for AutoGluon and TabPFN
    X_auto = X_raw.copy()

    return X_xgb, X_auto, y


def evaluate_model(y_true, y_pred, y_prob, train_time, inference_time, seed):
    """
    Compute evaluation metrics for a single model run.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like
        Predicted probabilities for positive class.
    train_time : float
        Training time in seconds.
    inference_time : float
        Inference time in seconds.
    seed : int
        Random seed used for this run.

    Returns
    -------
    dict
        Dictionary with evaluation metrics.
    """
    return {
        'seed': seed,
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'train_time': train_time,
        'inference_time': inference_time
    }


def summarize_results(results_df, model_name):
    """
    Print mean ± std summary of results across seeds.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with results across seeds.
    model_name : str
        Name of the model.
    """
    print(f'\n{model_name} Results (mean ± std across seeds):')
    for metric in ['accuracy', 'auc_roc', 'f1']:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        print(f'  {metric}: {mean:.4f} ± {std:.4f}')
    print(f'  train_time: {results_df["train_time"].mean():.4f}s')
    print(f'  inference_time: {results_df["inference_time"].mean():.4f}s')


def get_train_test_split(X, y, seed, test_size=0.2):
    """
    Perform stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    seed : int
        Random seed.
    test_size : float
        Proportion of data for test set.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def introduce_missing_values(X, missing_rate, seed):
    """
    Artificially introduce missing values into a feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    missing_rate : float
        Proportion of values to set as NaN (e.g., 0.05 for 5%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Feature matrix with missing values introduced.
    """
    X_missing = X.copy()
    np.random.seed(seed)
    mask = np.random.random(X_missing.shape) < missing_rate
    X_missing = X_missing.where(~mask, other=np.nan)
    return X_missing
