"""automl.py
AutoML pipeline module integrated with the existing DataCleaningAgent.

Features implemented (per project spec):
- Automatic problem detection (regression vs classification)
- Automatic model recommendation by dataset size
- Imbalance handling (SMOTE/oversampling) for classification
- Candidate model training (sklearn, xgboost, lightgbm, simple MLP)
- Hyperparameter tuning: GridSearchCV for small, Optuna for medium
- Evaluation and model selection with appropriate metrics
- Feature importance extraction for explainability
- Predictions/export and model persistence
- Basic HTML/text reporting and plot outputs

Notes / assumptions:
- If no target column is provided, a basic heuristic is used: look for common names
  ("target", "label", "y"); otherwise the last column is assumed the target.
- For large datasets (>500k rows), a sample is used for expensive tuning/training steps.

This module is intended to be imported by `main_agent.DataCleaningAgent`.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import joblib

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except Exception:
    SMOTE = None
    RandomOverSampler = None

try:
    import optuna
except Exception:
    optuna = None

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_target_column(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[str, str]:
    """Detect target column and problem type.

    Returns (target_column, problem_type) where problem_type is 'regression' or 'classification'.

    Heuristics:
    - If target_col provided and present in df, use it.
    - Otherwise look for common names: 'target','label','y'.
    - Otherwise use the last column.
    - If dtype is numeric and nunique() > 20 -> regression else classification.
    """
    if target_col and target_col in df.columns:
        col = target_col
    else:
        common = [c for c in df.columns if c.lower() in ('target', 'label', 'y')]
        if common:
            col = common[0]
        else:
            # fallback: choose last column
            col = df.columns[-1]

    series = df[col].dropna()
    nunique = series.nunique()
    dtype = series.dtype

    # Heuristic: numeric with many uniques -> regression
    if pd.api.types.is_numeric_dtype(dtype) and nunique > 20:
        problem = 'regression'
    else:
        problem = 'classification' if nunique <= 20 or not pd.api.types.is_numeric_dtype(dtype) else 'regression'

    logger.info(f"Detected target column='{col}' nunique={nunique} dtype={dtype} -> problem={problem}")
    return col, problem


def recommend_models(problem: str, n_rows: int) -> List[str]:
    """Recommend candidate models given problem type and dataset size."""
    if n_rows < 50_000:
        base = ['random_forest', 'xgboost', 'lightgbm', 'mlp']
    elif n_rows < 500_000:
        base = ['xgboost', 'lightgbm', 'mlp_small']
    else:
        base = ['lightgbm']

    if problem == 'regression':
        models = ['linear_regression'] + base
    else:
        models = ['logistic_regression'] + base

    logger.info(f"Recommended models for {problem} with {n_rows} rows: {models}")
    return models


def handle_imbalance(X: pd.DataFrame, y: pd.Series, strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series, bool]:
    """If classification and imbalance detected, apply SMOTE or oversampling.

    Returns (X_res, y_res, was_resampled)
    """
    if y is None or y.empty:
        return X, y, False

    value_counts = y.value_counts(normalize=True)
    if len(value_counts) <= 1:
        return X, y, False

    max_pct = value_counts.max()
    imbalance = max_pct > 0.7  # arbitrary threshold

    if not imbalance:
        return X, y, False

    logger.info(f"Imbalance detected in target: {value_counts.to_dict()}")

    if SMOTE is not None and strategy in ('auto', 'smote'):
        try:
            sm = SMOTE()
            X_res, y_res = sm.fit_resample(X, y)
            logger.info("Applied SMOTE resampling")
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res), True
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}")

    if RandomOverSampler is not None:
        ros = RandomOverSampler()
        X_res, y_res = ros.fit_resample(X, y)
        logger.info("Applied RandomOverSampler")
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res), True

    logger.warning("No resampling performed (imbalanced but imblearn not available)")
    return X, y, False


def _build_model(name: str, problem: str):
    """Return an instantiated model object (sklearn-like) given name."""
    name = name.lower()
    if name == 'linear_regression':
        return LinearRegression()
    if name == 'logistic_regression':
        return LogisticRegression(max_iter=1000)
    if name == 'random_forest':
        return RandomForestRegressor() if problem == 'regression' else RandomForestClassifier()
    if name == 'xgboost':
        if xgb is None:
            raise RuntimeError('xgboost not installed')
        return xgb.XGBRegressor(objective='reg:squarederror') if problem == 'regression' else xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    if name == 'lightgbm':
        if lgb is None:
            raise RuntimeError('lightgbm not installed')
        return lgb.LGBMRegressor() if problem == 'regression' else lgb.LGBMClassifier()
    if name == 'mlp' or name == 'mlp_small':
        if problem == 'regression':
            return MLPRegressor(hidden_layer_sizes=(100,))
        else:
            return MLPClassifier(hidden_layer_sizes=(100,))

    raise ValueError(f"Unknown model name: {name}")


def _default_param_grid(name: str, problem: str) -> Dict[str, list]:
    name = name.lower()
    if name in ('random_forest',):
        if problem == 'regression':
            return {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        else:
            return {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    if name == 'xgboost':
        return {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
    if name == 'lightgbm':
        return {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
    if name in ('linear_regression', 'logistic_regression'):
        return {}
    if name.startswith('mlp'):
        return {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
    return {}


from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

def _evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Evaluate regression model performance safely across sklearn versions."""
    try:
        # ✅ For newer scikit-learn versions
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # ⚙️ For older versions without 'squared' argument
        rmse = sqrt(mean_squared_error(y_true, y_pred))
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}



def _evaluate_classification(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = None
    try:
        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                auc = roc_auc_score(y_true, y_proba)
            else:
                # multiclass
                auc = roc_auc_score(y_true, y_proba, multi_class='ovo')
    except Exception:
        auc = None

    res = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
    if auc is not None:
        res['auc'] = float(auc)
    return res


def _fit_and_tune(model_name: str, model, X_train, y_train, dataset_size: str) -> Tuple[Any, Dict]:
    """Fit model and optionally tune hyperparameters. Returns fitted model and tuning info."""
    info = {'model_name': model_name}
    grid = _default_param_grid(model_name, 'regression' if hasattr(model, 'predict') and 'Regressor' in type(model).__name__ else 'classification')

    if dataset_size == 'small' and grid:
        try:
            gs = GridSearchCV(model, grid, cv=3, n_jobs=-1, scoring=None)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            info['tuning'] = {'method': 'grid', 'best_params': gs.best_params_}
            return best, info
        except Exception as e:
            logger.warning(f"GridSearch failed for {model_name}: {e}")

    if dataset_size == 'medium' and optuna is not None:
        # Simple optuna tune for n_estimators / learning_rate when applicable
        def _objective(trial):
            params = {}
            if model_name in ('xgboost', 'lightgbm'):
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
                params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
            elif model_name == 'random_forest':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            elif model_name.startswith('mlp'):
                params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-1)

            try:
                m = _build_model(model_name, 'regression' if hasattr(y_train, 'dtype') and np.issubdtype(y_train.dtype, np.number) else 'classification')
                m.set_params(**params)
                scores = cross_val_score(m, X_train, y_train, cv=3, scoring='neg_mean_squared_error' if np.issubdtype(y_train.dtype, np.number) else 'accuracy')
                return float(np.mean(scores))
            except Exception as e:
                raise

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(_objective, n_trials=20)
            best_params = study.best_params
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            info['tuning'] = {'method': 'optuna', 'best_params': best_params}
            return model, info
        except Exception as e:
            logger.warning(f"Optuna tuning failed for {model_name}: {e}")

    # Default: fit without tuning
    model.fit(X_train, y_train)
    info['tuning'] = {'method': 'none'}
    return model, info


def _extract_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    fi = {}
    try:
        if hasattr(model, 'feature_importances_'):
            vals = model.feature_importances_
            fi = dict(zip(feature_names, [float(x) for x in vals]))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            fi = dict(zip(feature_names, [float(abs(x)) for x in coef]))
    except Exception:
        fi = {}
    return fi


def run_automl_pipeline(df: pd.DataFrame,
                        output_dir: str = 'output',
                        target_col: Optional[str] = None,
                        sample_if_large: bool = True,
                        random_state: int = 42,
                        run_full_report: bool = True) -> Dict[str, Any]:
    """Run an end-to-end AutoML pipeline on a processed DataFrame.

    Returns a dictionary containing the best model, evaluation metrics, file paths, etc.
    """
    out = {'output_dir': output_dir}
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    col, problem = detect_target_column(df, target_col)
    X = df.drop(columns=[col])
    y = df[col]

    n_rows = len(df)
    if n_rows > 500_000 and sample_if_large:
        logger.info("Large dataset detected: sampling for AutoML steps")
        samp = df.sample(n=200_000, random_state=random_state)
        X = samp.drop(columns=[col])
        y = samp[col]
        dataset_size = 'large'
    elif n_rows >= 50_000:
        dataset_size = 'medium'
    else:
        dataset_size = 'small'

    candidates = recommend_models(problem, n_rows)

    # Handle imbalance (classification)
    was_resampled = False
    if problem == 'classification':
        X, y, was_resampled = handle_imbalance(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    trained_models = {}
    evaluations = {}
    feature_importances = {}

    for name in candidates:
        try:
            logger.info(f"Training candidate: {name}")
            model = _build_model(name, problem)
            fitted, info = _fit_and_tune(name, model, X_train, y_train, dataset_size)

            # Predict and evaluate
            y_pred = fitted.predict(X_test)
            y_proba = None
            if problem == 'classification' and hasattr(fitted, 'predict_proba'):
                try:
                    y_proba = fitted.predict_proba(X_test)
                except Exception:
                    y_proba = None

            if problem == 'regression':
                metrics = _evaluate_regression(y_test, y_pred)
            else:
                metrics = _evaluate_classification(y_test, y_pred, y_proba)

            trained_models[name] = fitted
            evaluations[name] = {'metrics': metrics, 'info': info}

            # Feature importance
            fi = _extract_feature_importance(fitted, X.columns.tolist())
            feature_importances[name] = fi

            # Save model artifact
            model_path = outdir / f"model_{name}.joblib"
            joblib.dump(fitted, model_path)
            evaluations[name]['model_path'] = str(model_path)

            logger.info(f"Completed training for {name} — metrics: {metrics}")

        except Exception as e:
            logger.exception(f"Failed training {name}: {e}")

    # Select best model
    best_name = None
    best_score = None
    for name, ev in evaluations.items():
        m = ev['metrics']
        if problem == 'regression':
            score = -m.get('rmse', float('inf'))  # lower RMSE better
        else:
            score = m.get('f1', m.get('accuracy', 0))

        if best_score is None or score > best_score:
            best_score = score
            best_name = name

    if best_name:
        out['best_model_name'] = best_name
        out['best_model_path'] = evaluations[best_name].get('model_path')
        out['best_model_metrics'] = evaluations[best_name]['metrics']

    # Save predictions for best model
    if best_name:
        best_model = trained_models[best_name]
        preds = best_model.predict(X_test)
        preds_df = pd.DataFrame({'actual': y_test.reset_index(drop=True), 'prediction': pd.Series(preds)})
        preds_path = outdir / 'predictions_best_model.csv'
        preds_df.to_csv(preds_path, index=False)
        out['predictions_path'] = str(preds_path)

    # Create simple reports
    if run_full_report:
        # Save evaluations to CSV
        eval_rows = []
        for name, ev in evaluations.items():
            row = {'model': name}
            row.update({f"metric_{k}": v for k, v in ev['metrics'].items()})
            row.update({'model_path': ev.get('model_path')})
            eval_rows.append(row)

        eval_df = pd.DataFrame(eval_rows)
        eval_path = outdir / 'model_evaluations.csv'
        eval_df.to_csv(eval_path, index=False)
        out['evaluations_csv'] = str(eval_path)

        # Feature importance plots for best model
        if best_name and feature_importances.get(best_name):
            fi = feature_importances[best_name]
            fi_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:30]
            names = [i[0] for i in fi_items]
            vals = [i[1] for i in fi_items]
            plt.figure(figsize=(8, max(4, len(names)*0.25)))
            plt.barh(names[::-1], vals[::-1])
            plt.xlabel('Importance')
            plt.title(f'Feature importance ({best_name})')
            fig_path = outdir / f'feature_importance_{best_name}.png'
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            out['feature_importance_plot'] = str(fig_path)

        # Simple HTML summary
        report_lines = []
        report_lines.append(f"AutoML Report\nBest model: {best_name}\n")
        if best_name:
            for k, v in out['best_model_metrics'].items():
                report_lines.append(f"{k}: {v}")

        report_text = "\n".join(report_lines)
        report_path = outdir / 'automl_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        out['report_path'] = str(report_path)

    return out
