# audit_tool/explainability.py

import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def compute_shap_values(model, X):
    """
    Compute SHAP values using the modern SHAP API.
    Automatically handles different model types and disables additivity check
    for tree-based classifiers when necessary.

    Parameters:
        model: A trained ML model (e.g., tree, linear, etc.)
        X: The input dataset (DataFrame) to explain

    Returns:
        shap_values: A shap.Explanation object
        X_numeric: The numeric version of X used for computation
    """

    # Ensure input is numeric and clean
    X_numeric = X.astype(float).reset_index(drop=True)

    # Build explainer using modern unified API
    explainer = shap.Explainer(model, X_numeric)

    # For models that are known to trigger additivity errors (e.g. RandomForestClassifier)
    tree_models_additivity_issue = (RandomForestClassifier,)

    if isinstance(model, tree_models_additivity_issue):
        shap_values = explainer(X_numeric, check_additivity=False)
    else:
        shap_values = explainer(X_numeric)

    return shap_values, X_numeric



def plot_shap_summary(shap_values, X, max_display=20):
    """
    Plot a SHAP summary plot, automatically handling binary classification.
    
    Parameters:
        shap_values: Output of TreeExplainer or SHAP Explainer
        X: Feature set used to compute SHAP values (numeric)
        max_display: Max number of features to show in the summary plot
    """
    # If shap_values is a list, assume it's a classification task and pick class 1
    if isinstance(shap_values, list) and len(shap_values) == 2:
        values_to_plot = shap_values[1]
    else:
        values_to_plot = shap_values

    shap.summary_plot(values_to_plot, X, max_display=max_display)
    plt.tight_layout()

def plot_model_feature_importance(model, feature_names, top_n=10, importance_type="gain", return_data=True):
    """
    Plot and return feature importances for tree-based models (RandomForest, XGBoost, LightGBM).
    Prints a message if the model does not support built-in feature importances.

    Parameters:
        model: Trained ML model
        feature_names: List of feature names
        top_n: Number of top features to plot
        importance_type: 'weight', 'gain', or 'cover' (for XGBoost)
        return_data: If True, returns a DataFrame with feature importances

    Returns:
        DataFrame with feature names and importances, sorted descending (if return_data=True)
        None if feature importances not available or return_data=False
    """

    df = None

    if hasattr(model, "feature_importances_"):
        # For RandomForest, GradientBoosting, LightGBM
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        top_df = df.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.title("Top Feature Importances")
        plt.bar(range(len(top_df)), top_df["importance"], align="center")
        plt.xticks(range(len(top_df)), top_df["feature"], rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.xlabel("Feature")
        plt.tight_layout()
        plt.show()

    elif isinstance(model, XGBClassifier):
        booster = model.get_booster()
        scores = booster.get_score(importance_type=importance_type)

        # Fill in zero for missing features
        all_scores = {f: scores.get(f, 0.0) for f in feature_names}
        df = pd.DataFrame(list(all_scores.items()), columns=["feature", "importance"])
        df = df.sort_values(by="importance", ascending=False)

        top_df = df.head(top_n)

        if top_df.empty:
            print("[INFO] No feature importances available from XGBoost booster.")
            return None

        plt.figure(figsize=(10, 6))
        plt.title(f"Top Feature Importances (XGBoost - {importance_type})")
        plt.bar(range(len(top_df)), top_df["importance"], align="center")
        plt.xticks(range(len(top_df)), top_df["feature"], rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.xlabel("Feature")
        plt.tight_layout()
        plt.show()

    else:
        print("[INFO] Feature importances not available for this model type.")
        return None

    if return_data:
        return df.reset_index(drop=True)
    else:
        return None

def plot_shap_waterfall(shap_values, instance_idx, model=None, X=None, y_true=None, class_idx=1):
    """
    Plot a SHAP waterfall plot for a single instance, with optional prediction and true label info.

    Parameters:
        shap_values: SHAP values (array or list of arrays)
        instance_idx: Index of the instance to explain
        model: Trained model (optional, used to get prediction)
        X: Feature matrix (optional, required if model is given)
        y_true: True labels (optional, for displaying true class)
        class_idx: Class index for binary classification (default: 1)
    """

    # Select SHAP values depending on classification/regression
    if isinstance(shap_values, list):
        if instance_idx >= len(shap_values[class_idx]):
            raise ValueError(f"Instance index {instance_idx} out of bounds.")
        shap_instance = shap_values[class_idx][instance_idx]
    else:
        if instance_idx >= len(shap_values):
            raise ValueError(f"Instance index {instance_idx} out of bounds.")
        shap_instance = shap_values[instance_idx]

    # Construct dynamic title
    title = f"SHAP Waterfall Plot — Instance {instance_idx}"

    if model is not None and X is not None:
        try:
            pred = model.predict_proba(X)[instance_idx, class_idx]
            title += f" | Prediction: {pred:.2f}"
        except AttributeError:
            try:
                pred = model.predict(X)[instance_idx]
                title += f" | Prediction: {pred:.2f}"
            except Exception:
                pass

    if y_true is not None:
        try:
            true_label = y_true.iloc[instance_idx] if hasattr(y_true, "iloc") else y_true[instance_idx]
            title += f" | True class: {true_label}"
        except Exception:
            pass

    # Plot
    shap.plots.waterfall(shap_instance, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()



def explain_instance_lime(
    model, X, instance_idx, y_true=None,
    explainer=None, categorical_keywords=None,
    class_names=["Non-Default", "Default"],
):
    """
    Generate and display a LIME explanation for a single instance.

    Parameters:
        model: Trained classification model with predict_proba
        X: Unscaled feature DataFrame (used for interpretability)
        instance_idx: Index of the instance to explain
        y_true: True labels (optional, for reporting)
        explainer: Optional pre-built LimeTabularExplainer
        categorical_keywords: Keywords to identify categorical features
        class_names: Class labels (used for LIME display)
        export_html: If True, saves the explanation as an HTML file
        html_path: Path to save the HTML file
    """

    if not 0 <= instance_idx < len(X):
        raise ValueError(f"Instance index {instance_idx} is out of bounds.")

    # Detect categorical columns
    if categorical_keywords is None:
        categorical_keywords = ['State', 'Occupation', 'Employment', 'Quarter', 'Category']

    cat_features = [
        X.columns.get_loc(col)
        for col in X.columns
        if any(kw in col for kw in categorical_keywords)
    ]

    # Create explainer if not provided
    if explainer is None:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=class_names,
            categorical_features=cat_features,
            mode='classification'
        )

    # Explain the instance
    exp = explainer.explain_instance(
        data_row=X.iloc[instance_idx].values,
        predict_fn=model.predict_proba
    )

    # Display summary info
    print(f"LIME Explanation for instance {instance_idx}")
    try:
        pred_proba = model.predict_proba(X)[instance_idx, 1]
        print(f"Model prediction (P=Default): {pred_proba:.2f}")
    except Exception:
        pass

    if y_true is not None:
        # Get actual label safely
        try:
            actual = y_true.iloc[instance_idx] if hasattr(y_true, "iloc") else y_true[instance_idx]
            if isinstance(actual, pd.Series):
                actual = actual.item()
            label_text = class_names[actual] if actual in [0, 1] else actual
            print(f"True label: {actual} ({label_text})")
        except Exception as e:
            print(f"[WARN] Could not parse true label: {e}")


    # Show explanation
    exp.show_in_notebook()

    return exp
