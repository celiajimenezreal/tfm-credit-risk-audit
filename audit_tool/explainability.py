# audit_tool/explainability.py

import shap
import pandas as pd
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

def compute_shap_values(model, X):
    """
    Compute SHAP values for a given model and dataset.
    Ensures X is numeric and has no object dtypes.
    """
    # Ensure numeric dtype
    X_numeric = X.astype(float).reset_index(drop=True)
    explainer = shap.Explainer(model, X_numeric)
    shap_values = explainer(X_numeric)
    return shap_values

def plot_shap_summary(shap_values, X, max_display=20):
    """
    Plot SHAP summary plot.
    """
    shap.summary_plot(shap_values, X, max_display=max_display)

def plot_shap_waterfall(shap_values, instance_idx):
    """
    Plot SHAP waterfall plot for a single instance.
    """
    if instance_idx >= len(shap_values):
        raise ValueError(f"Instance index {instance_idx} out of bounds.")
    shap.plots.waterfall(shap_values[instance_idx])

def plot_rf_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importances from a Random Forest model.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
    plt.tight_layout()
    plt.show()

def get_lime_explainer(X, categorical_features_keywords=None):
    """
    Create and return a LIME explainer.
    categorical_features_keywords: list of strings to identify categorical columns (e.g., ['State', 'Occupation'])
    """
    if categorical_features_keywords is None:
        categorical_features_keywords = ['BorrowerState', 'Occupation', 'EmploymentStatus', 'LoanOriginationQuarter', 'ListingCategory']

    categorical_features = [X.columns.get_loc(col) for col in X.columns 
                            if any(keyword in col for keyword in categorical_features_keywords)]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["Non-Default", "Default"],
        categorical_features=categorical_features,
        mode='classification'
    )

    return explainer

def plot_lime_explanation(explainer, X, model, instance_idx=0):
    """
    Plot LIME explanation for a specific instance.
    """
    exp = explainer.explain_instance(
        data_row=X.iloc[instance_idx].values,
        predict_fn=model.predict_proba
    )
    exp.show_in_notebook()

