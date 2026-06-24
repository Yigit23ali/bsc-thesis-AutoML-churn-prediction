"""
src/train.py
------------
Training functions for XGBoost, AutoGluon, and TabPFN models.

Author: Yigit Ali Uysal
Supervisor: Elias Dubbeldam
University of Amsterdam, 2026
"""

import time
import os
import shutil
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import get_train_test_split, evaluate_model


def train_xgboost(X, y, seeds, params=None):
    """
    Train XGBoost across multiple seeds and return results.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (one-hot encoded).
    y : pd.Series
        Target variable.
    seeds : list
        List of random seeds.
    params : dict, optional
        Hyperparameters. If None, uses default parameters.

    Returns
    -------
    pd.DataFrame
        Results across seeds.
    """
    from xgboost import XGBClassifier

    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = get_train_test_split(X.astype(float), y, seed)

        model_params = params or {}
        model_params['eval_metric'] = 'logloss'

        start = time.time()
        model = XGBClassifier(**model_params, random_state=seed)
        model.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start

        results.append(evaluate_model(y_test, y_pred, y_prob, train_time, inference_time, seed))

    return pd.DataFrame(results)


def tune_xgboost(X, y, n_trials=50, seed=0):
    """
    Tune XGBoost hyperparameters using Optuna.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (one-hot encoded).
    y : pd.Series
        Target variable.
    n_trials : int
        Number of Optuna trials.
    seed : int
        Random seed for train/test split during tuning.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    import optuna
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, _, y_train, _ = get_train_test_split(X.astype(float), y, seed)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'Best CV AUC: {study.best_value:.4f}')
    print(f'Best params: {study.best_params}')
    return study.best_params


def train_autogluon(X, y, seeds, time_limit=120, presets='high_quality',
                    num_stack_levels=None, num_bag_folds=None):
    """
    Train AutoGluon across multiple seeds and return results.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw categorical features).
    y : pd.Series
        Target variable.
    seeds : list
        List of random seeds.
    time_limit : int
        Training time limit in seconds.
    presets : str
        AutoGluon preset quality level.
    num_stack_levels : int, optional
        Number of stacking levels.
    num_bag_folds : int, optional
        Number of bagging folds.

    Returns
    -------
    pd.DataFrame
        Results across seeds.
    """
    from autogluon.tabular import TabularPredictor

    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = get_train_test_split(X, y, seed)

        train_data = X_train.copy()
        train_data['Exited'] = y_train.values

        save_path = f'agmodels/seed_{seed}'
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        fit_kwargs = {'time_limit': time_limit, 'presets': presets}
        if num_stack_levels is not None:
            fit_kwargs['num_stack_levels'] = num_stack_levels
        if num_bag_folds is not None:
            fit_kwargs['num_bag_folds'] = num_bag_folds

        start = time.time()
        predictor = TabularPredictor(label='Exited', path=save_path, verbosity=0)
        predictor.fit(train_data, **fit_kwargs)
        train_time = time.time() - start

        start = time.time()
        y_prob = predictor.predict_proba(X_test)[1]
        y_pred = predictor.predict(X_test)
        inference_time = time.time() - start

        results.append(evaluate_model(y_test, y_pred, y_prob, train_time, inference_time, seed))
        shutil.rmtree(save_path)

    return pd.DataFrame(results)


def train_tabpfn(X, y, seeds):
    """
    Train TabPFN (cloud API) across multiple seeds and return results.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw categorical features).
    y : pd.Series
        Target variable.
    seeds : list
        List of random seeds.

    Returns
    -------
    pd.DataFrame
        Results across seeds.
    """
    from tabpfn_client import TabPFNClassifier as CloudTabPFN

    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = get_train_test_split(X, y, seed)

        start = time.time()
        model = CloudTabPFN(random_state=seed)
        model.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        inference_time = time.time() - start

        results.append(evaluate_model(y_test, y_pred, y_prob, train_time, inference_time, seed))

    return pd.DataFrame(results)
