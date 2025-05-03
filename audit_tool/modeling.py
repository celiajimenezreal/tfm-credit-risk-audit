# audit_tool/modeling.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(random_state=42)
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
