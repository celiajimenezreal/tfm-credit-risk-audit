# reporting.py
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
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False

# --- Config paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports', 'explainability')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
TEMPLATE_FILE = 'explainability_report.html.j2'
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Jinja2 Template ---
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml'])
)
template = env.get_template(TEMPLATE_FILE)

# --- Utilidades ---
def plot_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def detect_outliers(df: pd.DataFrame, features: list, z_thresh: float = 3.0) -> dict:
    """Return percentage of outliers per feature based on z-score threshold."""
    stats = {}
    for f in features:
        col = df[f].dropna()
        z = (col - col.mean()) / col.std()
        stats[f] = float((abs(z) > z_thresh).mean())
    return stats

def compute_vif(df):
    if not STATS_MODELS_AVAILABLE:
        return pd.DataFrame()
    X = df.select_dtypes(include=[np.number]).dropna()
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return pd.DataFrame({'feature': X.columns, 'VIF': vif}).sort_values('VIF', ascending=False)

def check_calibration(model, X, y):
    proba = model.predict_proba(X.astype(float))[:, 1]
    return brier_score_loss(y, proba)

def sample_for_shap(X, max_samples=1000):
    return X.sample(max_samples, random_state=42) if len(X) > max_samples else X

def get_shap_explainer(model, X):
    if isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
        return shap.TreeExplainer(model), shap.TreeExplainer(model)(X)
    elif hasattr(model, 'coef_'):
        return shap.LinearExplainer(model, X), shap.LinearExplainer(model, X)(X)
    else:
        return shap.KernelExplainer(model.predict_proba, X), shap.KernelExplainer(model.predict_proba, X)(X)

def generate_report(model_name, model, X_train, X_test, y_test, instance_indices=None):
    instance_indices = instance_indices or [0, 10]
    suspicious_keywords = ['id', 'key', 'number', 'rating', 'score', 'date']

    shap_data = sample_for_shap(X_test)
    explainer, shap_vals = get_shap_explainer(model, shap_data)
    if hasattr(shap_vals, 'values') and shap_vals.values.ndim == 3:
        shap_summary = shap_vals[:, :, 1]
    else:
        shap_summary = shap_vals

    shap.summary_plot(shap_summary, shap_data, show=False)
    shap_summary_img = plot_to_base64()

    # Feature Importance (FI)
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
        plt.figure(figsize=(8, 5))
        plt.barh(fi_df['feature'][::-1][:15], fi_df['importance'][::-1][:15])
        plt.title(f'{model_name} Feature Importances')
        fi_img = plot_to_base64()
    except Exception:
        fi_df, fi_img = pd.DataFrame(), ''

    # Permutation importance
    perm_img, perm_df = '', pd.DataFrame()
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        perm_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': perm.importances_mean
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        plt.figure(figsize=(8, 5))
        plt.barh(perm_df['feature'][:15][::-1], perm_df['importance'][:15][::-1])
        plt.title("Permutation Importance")
        perm_img = plot_to_base64()
    except Exception:
        pass

    # PDP
    pdp_img, top_feat = '', fi_df['feature'][0] if not fi_df.empty else X_train.columns[0]
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(model, X_train, [top_feat], ax=ax)
        pdp_img = plot_to_base64()
        print(f"[INFO] PDP generated for '{top_feat}'")
    except Exception as e:
        print(f"[WARNING] PDP failed for '{top_feat}': {e}")

    # Histogramas SHAP por top features
    histograms = []
    shap_array = shap_summary.values if hasattr(shap_summary, 'values') else shap_summary
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    for i in top_idx:
        f = X_train.columns[i]
        plt.figure()
        plt.hist(X_train[f], bins=30, alpha=0.7, color='gray')
        plt.xlabel(f)
        plt.title(f'Distribution of {f}')
        histograms.append({'name': f, 'b64': plot_to_base64()})

    # SHAP Waterfalls y LIME
    waterfalls = []
    lime = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(c) for c in sorted(y_test.unique())],
        mode='classification')
    lime_html_parts = []
    shap_top = []
    for idx in instance_indices:
        plt.figure()
        shap.plots.waterfall(shap_summary[idx], show=False)
        waterfalls.append({'idx': idx, 'img': plot_to_base64()})

        exp = lime.explain_instance(X_test.iloc[idx].values, model.predict_proba)
        lime_html_parts.append(exp.as_html())
        shap_top.append([X_train.columns[i] for i in np.argsort(np.abs(shap_summary[idx].values))[::-1][:3]])

    lime_html = '\n<hr/>\n'.join(lime_html_parts)

    # Comparación de rankings (SHAP vs FI vs Permutation)
    comp_table = []
    if not fi_df.empty and not perm_df.empty:
        shap_rank = np.argsort(-mean_abs_shap)
        perm_rank = {f: i for i, f in enumerate(perm_df['feature'])}
        fi_rank = {f: i for i, f in enumerate(fi_df['feature'])}
        for i in range(10):
            feat = X_train.columns[shap_rank[i]]
            comp_table.append({
                'feature': feat,
                'shap': i + 1,
                'fi': fi_rank.get(feat, '-'),
                'perm': perm_rank.get(feat, '-'),
                'consistency': '✔️' if fi_rank.get(feat, 999) < 5 and perm_rank.get(feat, 999) < 5 else '⚠️'
            })

    # Recomendaciones y Conclusiones
    recommendations, conclusions = [], []

    # --- Análisis PDP ---
    if pdp_img:
        conclusions.append(f'The Partial Dependence Plot (PDP) for "{top_feat}" illustrates its marginal effect on the model’s predictions.')
    else:
        conclusions.append(f'Partial Dependence Plot (PDP) could not be generated for "{top_feat}". This may indicate incompatibility with the estimator.')

    # --- Feature Importance ---
    if not fi_df.empty:
        top1 = fi_df.iloc[0]
        conclusions.append(f'The feature "{top1.feature}" has the highest model importance ({top1.importance:.3f}), indicating it strongly drives model predictions.')
    else:
        conclusions.append("Feature importance could not be extracted from the model.")

    # --- SHAP-based conclusions ---
    shap_array = shap_summary.values if hasattr(shap_summary, 'values') else shap_summary
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-6:][::-1]
    for i in top_idx:
        val = float(mean_abs_shap[i])
        std = float(np.abs(shap_array[:, i]).std())
        fname = X_train.columns[i]
        conclusions.append(f'SHAP shows feature "{fname}" has high average contribution ({val:.4f}) with std={std:.4f}.')
        if std > 0.2 * val:
            conclusions.append(f'This indicates potential non-linear or interaction effects for "{fname}".')

    # SHAP instance overlap
    if len(instance_indices) >= 2:
        overlap = set(shap_top[0]).intersection(set(shap_top[1]))
        conclusions.append(f'SHAP explanations for instances {instance_indices[0]} and {instance_indices[1]} share {len(overlap)} common top features.')
        if len(overlap) <= 1:
            conclusions.append('Low agreement between SHAP explanations across instances may indicate high local sensitivity.')

    # Waterfalls & LIME
    if len(waterfalls) > 0:
        conclusions.append('Local SHAP waterfall plots reveal the additive impact of features on individual predictions.')
    if lime_html_parts:
        conclusions.append('LIME explanations complement SHAP by providing linear approximations for selected instances.')

    # Comparativa FI vs Permutation
    if not fi_df.empty and not perm_df.empty:
        perm_sorted_idx = perm_df['feature'].tolist()
        fi_rank = {f: i for i, f in enumerate(fi_df['feature'])}
        for f in perm_sorted_idx[:5]:
            if f in fi_rank and fi_rank[f] > 5:
                conclusions.append(f'Permutation importance highlights "{f}" as influential, despite its lower ranking in model-based FI. Consider reassessing this variable.')

    # --- Recomendaciones ---
    suspicious_feats = [f for f in X_train.columns if any(kw.lower() in f.lower() for kw in ['id', 'key', 'number', 'rating', 'score', 'date'])]
    if suspicious_feats:
        recommendations.append('Potential data leakage risk: check variables ' + ', '.join(suspicious_feats) + '.')

    if STATS_MODELS_AVAILABLE:
        vif_df = compute_vif(X_train)
        high_vif = vif_df[vif_df['VIF'] > 5]
        if not high_vif.empty:
            recommendations.append('High multicollinearity (VIF > 5) in: ' + ', '.join(high_vif['feature']))

    brier = check_calibration(model, X_test, y_test)
    if brier > 0.1:
        recommendations.append(f'Brier score = {brier:.3f}. Consider Platt Scaling or Isotonic Regression to improve calibration.')
    else:
        recommendations.append(f'Brier score = {brier:.3f}. Model is well calibrated.')

    # Outlier detection for top features
    top_feats = fi_df['feature'].tolist()[:5] if not fi_df.empty else X_train.columns[:5]
    outlier_stats = detect_outliers(X_train, top_feats)
    for feat, pct in outlier_stats.items():
        if pct > 0.01:
            recommendations.append(f'{pct*100:.1f}% of values for "{feat}" are outliers (>3σ). Consider log-transforming or winsorizing.')

    # Render final report
    # Verificaciones por ausencia de imágenes y mensajes alternativos
    if not fi_img:
        fi_img = ''  # Asegurar variable válida
        conclusions.append("Model does not expose built-in feature importance; this may limit interpretability.")
    if not perm_img:
        perm_img = ''
        conclusions.append("Permutation importance could not be computed. Model may lack support for this estimator.")
    if not pdp_img:
        pdp_img = ''
        conclusions.append("Partial Dependence Plot was not generated due to incompatibility or estimator constraints.")
    if not histograms:
        conclusions.append("Histograms of top features could not be rendered.")
    if not waterfalls:
        conclusions.append("SHAP waterfall plots could not be generated.")
    if not lime_html:
        conclusions.append("LIME explanations could not be generated due to model or data constraints.")

    # Conclusión adicional basada en número total de features relevantes
    if not fi_df.empty and len(fi_df) < 5:
        conclusions.append("Model relies heavily on a small set of variables. Consider assessing risk of overfitting or bias.")

    if not fi_df.empty and not perm_df.empty:
        common_top_feats = set(fi_df['feature'][:5]).intersection(set(perm_df['feature'][:5]))
        if len(common_top_feats) >= 3:
            conclusions.append(f'Model shows high agreement across FI and Permutation for top variables: {", ".join(common_top_feats)}.')
        elif len(common_top_feats) <= 1:
            conclusions.append("Low agreement between FI and Permutation indicates unstable variable importance. Review model robustness.")

    # Render final report (sin importar si se pudo generar cada parte)
    html = template.render(
        model_name=model_name,
        shap_summary_img=shap_summary_img,
        fi_plot=fi_img,
        feature_table=fi_df.to_dict(orient='records'),
        waterfalls=waterfalls,
        lime_explanations=lime_html,
        pdp_img=pdp_img,
        perm_img=perm_img,
        histograms=histograms,
        recommendations=recommendations,
        conclusions=conclusions,
        comparison_table=comp_table
    )

    out_path = os.path.join(REPORT_DIR, f"{model_name}_explainability_report.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report saved to {out_path}")
