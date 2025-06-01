# audit_tool/modeling.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print("Best Random Forest params:", grid_search.best_params_)
    return grid_search.best_estimator_


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

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

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
