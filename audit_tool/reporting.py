import os
import base64
import io
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular as lime_tabular
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
import joblib
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False
    print('[WARNING] statsmodels not available: VIF checks will be skipped.')
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports', 'explainability')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
TEMPLATE_FILE = 'explainability_report.html.j2'

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml'])
)
template = env.get_template(TEMPLATE_FILE)


def plot_to_base64():
    """
    Convert current matplotlib figure to a base64-encoded PNG string.
    """
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    return base64.b64encode(img_bytes).decode('utf-8')


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for numeric features."""
    if not STATS_MODELS_AVAILABLE:
        return pd.DataFrame()
    X = df.select_dtypes(include=[np.number]).dropna()
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return pd.DataFrame({'feature': X.columns, 'VIF': vif}).sort_values('VIF', ascending=False)


def check_calibration(model, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute Brier score for model probabilities on dataset."""
    proba = model.predict_proba(X.astype(float))[:, 1]
    return brier_score_loss(y, proba)


def detect_outliers(df: pd.DataFrame, features: list, z_thresh: float = 3.0) -> dict:
    """Return percentage of outliers per feature based on z-score threshold."""
    stats = {}
    for f in features:
        col = df[f].dropna()
        z = (col - col.mean()) / col.std()
        stats[f] = float((abs(z) > z_thresh).mean())
    return stats


def sample_for_shap(X: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
    """Random sample of dataset for global SHAP without altering original X."""
    if len(X) <= max_samples:
        return X
    return X.sample(max_samples, random_state=42)


def generate_report(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    instance_indices=None,
    suspicious_keywords: list = None
) -> None:
    """
    Generate an HTML explainability report for a model, including automated checks
    and recommendations, with optimized SHAP sampling for speed.
    """
    if instance_indices is None:
        instance_indices = [0, 10]
    if suspicious_keywords is None:
        suspicious_keywords = ['id', 'key', 'number', 'rating', 'score', 'date']

    # --- Global SHAP Summary (sampled) ---
    X_shap = sample_for_shap(X_test)
    # Unified SHAP Explainer (fast and produces Explanation objects)
    shap_explainer = shap.Explainer(model, X_shap.astype(float))
    shap_exp = shap_explainer(X_shap.astype(float))
    # Handle binary classification multi-dimensional outputs
    if hasattr(shap_exp, 'values') and shap_exp.values.ndim == 3:
        # select class 1 explanations
        shap_exp_summary = shap_exp[:, :, 1]
    else:
        shap_exp_summary = shap_exp
    # Plot SHAP summary
    shap.summary_plot(shap_exp_summary, X_shap, show=False)
    shap_summary_img = plot_to_base64()

    # --- Feature Importances ---
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).head(15)
        plt.figure(figsize=(8, 5))
        plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
        plt.xlabel('Importance')
        plt.title(f'{model_name} Feature Importances')
        fi_img = plot_to_base64()
    except Exception:
        fi_df = pd.DataFrame()
        fi_img = ''

    # --- Local SHAP Waterfalls ---
    waterfalls = []
    for idx in instance_indices:
        plt.figure()
        # select single-instance Explanation using summary layer
        inst = shap_exp_summary[idx]
        shap.plots.waterfall(inst, show=False)
        plt.title(f'Instance {idx}')
        waterfalls.append({'idx': idx, 'img': plot_to_base64()})

    # --- LIME Explanations ---
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(c) for c in sorted(y_test.unique())],
        mode='classification'
    )
    lime_html_parts = []
    for idx in instance_indices:
        exp = explainer_lime.explain_instance(X_test.iloc[idx].values, model.predict_proba)
        lime_html_parts.append(exp.as_html())
    lime_html = '\n<hr/>\n'.join(lime_html_parts)

    # --- Automated Recommendations ---
    recommendations = []
    # Suspicious features
    suspicious_feats = [f for f in X_train.columns if any(kw.lower() in f.lower() for kw in suspicious_keywords)]
    if suspicious_feats:
        recommendations.append(
            'Review or drop potential leakage features: ' + ', '.join(suspicious_feats)
        )

    # 1) Multicollinearity
    if STATS_MODELS_AVAILABLE:
        vif_df = compute_vif(X_train)
        high_vif = vif_df[vif_df['VIF'] > 5]
        if not high_vif.empty:
            recommendations.append(
                'High multicollinearity (VIF > 5) detected in: ' + ', '.join(high_vif['feature'].tolist()) + \
                '. Consider PCA or removing features.'
            )
    
    # 2) Calibration
    brier = check_calibration(model, X_test, y_test)
    if brier > 0.1:
        recommendations.append(
            f'Brier score = {brier:.3f}. Consider Platt Scaling or Isotonic Regression for calibration.'
        )
    else:
        recommendations.append(f'Brier score = {brier:.3f}. Calibration looks good.')

    # 3) Outliers
    top_feats = fi_df['feature'].tolist()[:5] if not fi_df.empty else X_train.columns.tolist()[:5]
    outlier_stats = detect_outliers(X_train, top_feats)
    for feat, pct in outlier_stats.items():
        if pct > 0.01:
            recommendations.append(
                f'{pct*100:.1f}% of values for "{feat}" are outliers (>3σ). Consider winsorizing or log-transform.'
            )

    # Render HTML
    html_content = template.render(
        model_name=model_name,
        shap_summary_img=shap_summary_img,
        fi_plot=fi_img,
        feature_table=fi_df.to_dict(orient='records'),
        waterfalls=waterfalls,
        lime_explanations=lime_html,
        recommendations=recommendations
    )

    # Save report
    report_path = os.path.join(REPORT_DIR, f'{model_name}_explainability_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Report saved to {report_path}')


if __name__ == '__main__':
    # Load data
    X_train = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'prosperloan', 'X_train_lasso.csv'))
    y_train = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'prosperloan', 'y_train_lasso.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'prosperloan', 'X_test_lasso.csv'))
    y_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'prosperloan', 'y_test_lasso.csv')).squeeze()

    # Load models
    models = {
        'logistic_regression': joblib.load(os.path.join(PROJECT_ROOT, 'models', 'prosperloan', 'logisticregression.pkl')),
        'random_forest': joblib.load(os.path.join(PROJECT_ROOT, 'models', 'prosperloan', 'randomforest.pkl')),
        'xgboost': joblib.load(os.path.join(PROJECT_ROOT, 'models', 'prosperloan', 'xgboost.pkl'))
    }

    for name, mdl in models.items():
        generate_report(name, mdl, X_train, X_test, y_test)
