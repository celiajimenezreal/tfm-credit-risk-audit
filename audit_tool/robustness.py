"""
audit_tool.robustness module
----------------------------

Provides functions to evaluate model robustness against various
perturbations and adversarial attacks using the Adversarial Robustness
Toolbox (ART). Includes utilities for loading models, generating
adversarial examples (FGSM, PGD), adding noise, boundary testing,
label flipping, safe prediction, and reporting results with confusion
matrices.

Key Functions:
    • ensure_dir: Create directories if they do not exist
    • load_model: Load and wrap models for ART attacks (Sklearn or XGBoost)
    • apply_fgsm / apply_pgd: Generate adversarial samples with FGSM and PGD
    • add_gaussian_noise: Inject Gaussian noise into inputs
    • boundary_testing: Set features to min/max bounds randomly
    • flip_labels: Randomly flip a fraction of labels
    • safe_predict: Standardize prediction output format
    • evaluate_and_plot: Compute metrics and save confusion matrix
    • run_robustness_tests: Execute full suite of robustness evaluations

Dependencies:
    • numpy
    • pandas
    • matplotlib
    • scikit-learn
    • joblib
    • adversarial-robustness-toolbox (art)

Typical Usage:
    from audit_tool.robustness import run_robustness_tests
    run_robustness_tests(
        model_name='xgb',
        model_path='models/xgb.joblib',
        model_type='xgb',
        X=X_test.values,
        y=y_test.values
    )

Author:
    Celia Jiménez Real, University of Navarra – TFM Model Audit

Date:
    2025-07-05
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.estimators.classification.xgboost import XGBoostClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from sklearn.base import ClassifierMixin
from sklearn.utils import shuffle
from joblib import load
from typing import Union

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_model(path: str, model_type: str):
    model = load(path)
    if model_type == 'xgb':
        return XGBoostClassifier(model=model, nb_classes=2, clip_values=(0.0, 1.0))
    else:
        return SklearnClassifier(model=model)

def apply_fgsm(classifier, X, eps=0.1):
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    X_adv = attack.generate(X)
    return X_adv

def apply_pgd(classifier, X, eps=0.1, eps_step=0.05, max_iter=20):
    attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps_step, max_iter=max_iter)
    X_adv = attack.generate(X)
    return X_adv

def add_gaussian_noise(X, mean=0.0, std=0.1):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def boundary_testing(X, bounds=(0.0, 1.0)):
    X_test = X.copy()
    for i in range(X.shape[1]):
        if np.random.rand() > 0.5:
            X_test[:, i] = bounds[0]  # min value
        else:
            X_test[:, i] = bounds[1]  # max value
    return X_test

def flip_labels(y, flip_fraction=0.1):
    y_flipped = y.copy()
    n_flip = int(len(y) * flip_fraction)
    indices = np.random.choice(len(y), size=n_flip, replace=False)
    unique_classes = np.unique(y)
    for i in indices:
        other_classes = unique_classes[unique_classes != y[i]]
        y_flipped[i] = np.random.choice(other_classes)
    return y_flipped

def safe_predict(clf, X):
    y_pred = clf.predict(X)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        return np.argmax(y_pred, axis=1)
    elif y_pred.ndim > 1 and y_pred.shape[1] == 1:
        return (y_pred > 0.5).astype(int).ravel()
    else:
        return y_pred

def evaluate_and_plot(model_name: str, y_true, y_pred, title: str, save_path: str):
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay

    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

    return df_report

def run_robustness_tests(model_name: str, model_path: str, model_type: str,
                          X: np.ndarray, y: np.ndarray,
                          output_dir: str = "outputs/reports/robustness/figures"):

    print(f"Running robustness tests for: {model_name}")

    clf = load_model(model_path, model_type)
    original_preds = safe_predict(clf, X)

    # FGSM and PGD only for models that support gradients
    if model_name in ["xgb", "logreg"]:
        try:
            # FGSM
            fgsm_X = apply_fgsm(clf, X)
            fgsm_preds = safe_predict(clf, fgsm_X)
            evaluate_and_plot(model_name, y, fgsm_preds, "FGSM Attack",
                              f"{output_dir}/{model_name}/fgsm_confmat.png")

            # PGD
            pgd_X = apply_pgd(clf, X)
            pgd_preds = safe_predict(clf, pgd_X)
            evaluate_and_plot(model_name, y, pgd_preds, "PGD Attack",
                              f"{output_dir}/{model_name}/pgd_confmat.png")
        except Exception as e:
            print(f"Gradient-based attacks not supported for {model_name}: {e}")

    # Gaussian noise
    noisy_X = add_gaussian_noise(X)
    noisy_preds = safe_predict(clf, noisy_X)
    evaluate_and_plot(model_name, y, noisy_preds, "Gaussian Noise Injection",
                      f"{output_dir}/{model_name}/noise_confmat.png")

    # Boundary testing
    bound_X = boundary_testing(X)
    bound_preds = safe_predict(clf, bound_X)
    evaluate_and_plot(model_name, y, bound_preds, "Boundary Testing",
                      f"{output_dir}/{model_name}/boundary_confmat.png")

    # Label flipping
    flipped_y = flip_labels(y)
    original_preds_yflip = safe_predict(clf, X)
    evaluate_and_plot(model_name, flipped_y, original_preds_yflip, "Label Flipping",
                      f"{output_dir}/{model_name}/label_flipping_confmat.png")

    print(f"Finished robustness tests for: {model_name}")