"""
audit_tool.modeling module
---------------------------

Provides functions for training, tuning, evaluating, and persisting
machine learning classifiers including Logistic Regression, Random Forest,
and XGBoost. Supports manual grid tuning, randomized search, early stopping,
pipeline construction with preprocessing, and model evaluation metrics.

Key Functions:
    • train_logistic_regression: Simple Logistic Regression training
    • train_logistic_regression_tuned: Pipeline + randomized hyperparameter search
    • train_random_forest / train_random_forest_tuned: RF training with optional GridSearchCV
    • train_xgboost / tune_xgboost / train_xgboost_tuned: XGBoost training and tuning
    • evaluate_model: Generate classification report and ROC AUC
    • evaluate_with_threshold: Custom threshold evaluation for top percentile
    • save_model / load_model: Persist and load trained models

Dependencies:
    • pandas
    • numpy
    • scikit-learn
    • xgboost
    • joblib
    • scipy

Typical Usage:
    from audit_tool.modeling import train_xgboost_tuned, evaluate_model
    best_model, results_df = train_xgboost_tuned(X_train, y_train)
    report, auc = evaluate_model(best_model, X_test, y_test)

Author:
    Celia Jiménez Real, University of Navarra – TFM Model Audit

Date:
    2025-07-05
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, ParameterGrid, RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train, X_val, y_val):
    """
    Manually tune XGBoost by testing a grid of hyperparameters.
    Returns the model with the best AUC on validation data.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    best_auc = 0
    best_model = None
    best_params = {}

    for n_estimators in [100, 200, 300]:
        for max_depth in [3, 5, 7]:
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)

            print(f"n_estimators={n_estimators}, max_depth={max_depth}, AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

    print(f"\nBest XGBoost AUC: {best_auc:.4f} with params: {best_params}")
    return best_model

def train_logistic_regression_tuned(X_train, y_train, cv=5, n_iter=30):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('clf', LogisticRegression(max_iter=1000, solver='saga', tol=1e-3))
    ])

    param_dist = {
        'clf__penalty':     ['l1', 'l2', 'elasticnet'],
        'clf__C':           loguniform(1e-3, 1e2),
        'clf__l1_ratio':    [0.2, 0.5, 0.8],     
        'clf__class_weight':['balanced', None],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rand = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    rand.fit(X_train, y_train)

    print("Mejores parámetros:", rand.best_params_)
    print("Mejor AUC en CV:", rand.best_score_)
    return rand.best_estimator_

def train_random_forest_tuned(X_train, y_train,
                               test_size=0.2,
                               random_state=42,
                               param_grid=None,
                               cv_splits=5,
                               scoring='roc_auc',
                               class_weight='balanced'):
    """
    Trains a Random Forest using GridSearchCV with an internal train/validation split.
    Prints the best hyperparameters and the best CV AUC score.

    Parameters:
    - X_train, y_train: Complete training dataset
    - test_size: Proportion for internal validation split (default=0.2)
    - random_state: Random seed for reproducibility
    - param_grid: Dictionary of hyperparameters for GridSearchCV
    - cv_splits: Number of folds in cross-validation
    - scoring: Metric for refitting in GridSearchCV
    - class_weight: Class weight parameter for the classifier

    Returns:
    - best_rf: The Random Forest estimator with optimal hyperparameters
    - grid_search: The complete GridSearchCV object containing CV results
    """
    # Split data into training and internal validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        stratify=y_train,
        test_size=test_size,
        random_state=random_state
    )

    # Use default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [5, 10, 20]
        }

    # Configure cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    base_rf = RandomForestClassifier(class_weight=class_weight, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring=scoring,
        refit=scoring,
        cv=cv,
        return_train_score=True,
        n_jobs=-1
    )

    # Fit the model on the training split
    grid_search.fit(X_tr, y_tr)

    # Retrieve and print best results
    best_rf = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV AUC score:", grid_search.best_score_)


    return best_rf, grid_search

def train_xgboost_tuned(
    X_train, y_train,
    param_grid=None,
    val_size: float = 0.2,
    random_state: int = 42,
    early_stopping_rounds: int = 50
):
    """
    Busca el mejor XGBClassifier sobre un grid dado, usando 
    early stopping en un split interno (train/validation).
    
    Devuelve:
      - best_model: el XGBClassifier con mejor AUC en validación interna.
      - results_df: DataFrame con columnas de hiperparámetros + 'val_auc'.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth':    [3, 5, 7],
            'learning_rate':[0.05, 0.1],
            'reg_alpha':    [0, 0.1, 1],
            'reg_lambda':   [1, 10]
        }
    
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train,
        stratify=y_train,
        test_size=val_size,
        random_state=random_state
    )
    
    best_auc    = 0.0
    best_model  = None
    records     = []
    
    for params in ParameterGrid(param_grid):
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            **params
        )
        # Early stopping
        model.fit(
            X_tr_sub, y_tr_sub,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        # AUC in validation
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        rec = params.copy()
        rec['val_auc'] = auc
        records.append(rec)
        
        if auc > best_auc:
            best_auc   = auc
            best_model = model
    
    results_df = pd.DataFrame.from_records(records)
    results_df = results_df.sort_values('val_auc', ascending=False).reset_index(drop=True)
    
    print("→ Mejor AUC en validación interna:", best_auc.round(4))
    print("→ Mejores parámetros:\n", results_df.iloc[0].drop('val_auc').to_dict())
    
    return best_model, results_df

def train_random_forest(X_train, y_train, class_weight=None):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return classification report and AUC score.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    return report, auc


def evaluate_with_threshold(model, X, y, top_pct=0.15):
  
    probs = model.predict_proba(X)[:, 1]
    threshold = np.percentile(probs, 100 - 100 * top_pct)
    preds = (probs >= threshold).astype(int)

    print(f"Threshold (top {int(top_pct*100)}%): {threshold:.4f}")
    print("AUC-ROC:", roc_auc_score(y, probs).round(3))
    print(classification_report(y, preds, digits=3))


def save_model(model, filename):
    """
    Save a trained model to disk.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a saved model from disk.
    """
    return joblib.load(filename)
