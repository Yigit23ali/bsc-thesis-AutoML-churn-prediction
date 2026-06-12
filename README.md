# bsc-thesis-AutoML-churn-prediction
Comparing Traditional ML, AutoML and Tabular Foundation Models for Bank Customer Churn Prediction
# Bank Customer Churn Prediction: Comparing ML, AutoML and Tabular Foundation Models

This repository contains the code and experiments for my BSc Business Analytics thesis at the University of Amsterdam.

## Project Overview

This study compares three machine learning approaches for bank customer churn prediction:
- **XGBoost** — traditional gradient boosting baseline (default and tuned with Optuna)
- **AutoGluon** — AutoML framework with ensemble stacking
- **TabPFN** — Tabular Foundation Model with zero-shot inference (cloud API)

Models are evaluated on predictive performance (AUC-ROC, F1-score), computational efficiency, robustness to missing data, sensitivity to dataset size, and interpretability.

## Dataset

The dataset used is the [Bank Customer Churn dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) from Kaggle. It contains 10,000 records with 14 features. Place the file `Churn_Modelling.csv` in the root directory before running the notebooks.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yigit23ali/bsc-thesis-churn-prediction.git
cd bsc-thesis-churn-prediction
```

2. Create a conda environment:
```bash
conda create -n thesis python=3.10
conda activate thesis
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For TabPFN cloud API, login with:
```python
from tabpfn_client import init
init()
```

## Code Structure

The experiments are organized into separate notebooks:

| Notebook | Description |
|---|---|
| `1_data_exploration.ipynb` | Dataset overview, class distribution, feature visualizations |
| `2_baseline_models.ipynb` | XGBoost (default + tuned), AutoGluon, TabPFN baseline comparison with statistical significance testing |
| `3_autogluon_analysis.ipynb` | AutoGluon configuration experiments: preset quality, stack levels, bagging folds |
| `4_tabpfn_analysis.ipynb` | TabPFN dataset size experiment and missing value robustness |
| `5_feature_importance.ipynb` | Permutation-based feature importance for all three models |

All figures are saved to the `figures/` directory.

## Results Summary

| Model | AUC-ROC | F1-Score | Train Time (s) | Inference Time (s) |
|---|---|---|---|---|
| XGBoost (Default) | 0.8373 ± 0.0114 | 0.5678 ± 0.0236 | 0.12 | 0.003 |
| XGBoost (Tuned) | 0.8664 ± 0.0069 | 0.5702 ± 0.0098 | 0.33 | 0.003 |
| AutoGluon | 0.8642 ± 0.0067 | 0.5861 ± 0.0172 | 124.93 | 0.61 |
| TabPFN | 0.8750 ± 0.0051 | 0.6079 ± 0.0097 | 4.51 | 5.86 |

TabPFN achieved the highest AUC-ROC and F1-score. Paired t-tests confirmed that TabPFN significantly outperforms both XGBoost (p=0.0014) and AutoGluon (p=0.0005). Tuned XGBoost and AutoGluon were not significantly different (p=0.2464).

## Requirements

Key packages used:
- Python 3.10
- xgboost==3.2.0
- autogluon==1.5.0
- tabpfn==8.0.3
- tabpfn-client
- optuna
- shap
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Author

Yiğit Ali Uysal — BSc Business Analytics, University of Amsterdam
Supervisor: Elias Dubbeldam
