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
from audit_tool.robustness import (
    load_model, apply_fgsm, apply_pgd, add_gaussian_noise,
    boundary_testing, flip_labels, safe_predict, evaluate_and_plot, ensure_dir
)
from sklearn.metrics import accuracy_score, f1_score

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

REPORT_DIR_2 = os.path.join(PROJECT_ROOT, 'outputs', 'reports', 'robustness')
FIGURES_DIR_2 = os.path.join(REPORT_DIR_2, 'figures')
TEMPLATE_FILE_2 = 'robustness_report.html.j2'
os.makedirs(REPORT_DIR_2, exist_ok=True)

# --- Jinja2 Template ---
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml'])
)
template = env.get_template(TEMPLATE_FILE)
template_2 = env.get_template(TEMPLATE_FILE_2)

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

def generate_explainability_report(model_name, model, X_train, X_test, y_test, instance_indices=None):
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

def generate_robustness_report(model_name: str, model_path: str, model_type: str,
                                X: np.ndarray, y: np.ndarray, feature_names: list) -> None:
    clf = load_model(model_path, model_type)

    conclusions, recommendations, tests_summary = [], [], []

    def image_to_base64(path: str) -> str:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def plot_top_deltas(X_orig, X_adv, test_name):
        delta = np.abs(X_adv - X_orig)
        mean_delta = delta.mean(axis=0)
        df_delta = pd.DataFrame({'feature': feature_names, 'mean_delta': mean_delta})
        df_delta = df_delta.sort_values('mean_delta', ascending=False).head(10)
        plt.figure(figsize=(8, 4))
        plt.barh(df_delta['feature'][::-1], df_delta['mean_delta'][::-1])
        plt.title(f'Top Modified Features: {test_name}')
        plt.tight_layout()
        img_path = os.path.join(FIGURES_DIR_2, model_name, f"{test_name}_delta.png")
        ensure_dir(os.path.dirname(img_path))
        plt.savefig(img_path)
        plt.close()
        return image_to_base64(img_path), df_delta

    def test_block(name, X_mod, y_mod, desc, explain, base_acc=None):
        y_pred = safe_predict(clf, X_mod)
        acc = accuracy_score(y_mod, y_pred)
        f1 = f1_score(y_mod, y_pred)
        img_path = os.path.join(FIGURES_DIR_2, model_name, f"{name}.png")
        ensure_dir(os.path.dirname(img_path))
        df_report = evaluate_and_plot(model_name, y_mod, y_pred, name, img_path)

        test_entry = {
            'name': name,
            'description': desc,
            'explanation': explain,
            'accuracy': acc,
            'f1_score': f1,
            'img_conf': image_to_base64(img_path),
            'conclusions': [],
            'recommendations': []
        }

        # Solo generamos gráfico de perturbaciones si X fue modificado
        if name not in ["Baseline Performance", "Label Noise Simulation"]:
            delta_img, delta_df = plot_top_deltas(X, X_mod, name)
            test_entry['img_pert'] = delta_img
            test_entry['top_deltas'] = delta_df.head(5).to_dict(orient='records')

            # Alta sensibilidad a alguna variable
            if delta_df['mean_delta'].iloc[0] > 0.2:
                msg = f"Model shows high sensitivity to feature: '{delta_df['feature'].iloc[0]}' during {name}."
                conclusions.append(msg)
                test_entry['conclusions'].append(msg)

        # A partir de aquí, aplicamos diagnósticos a todo excepto baseline
        if name != "Baseline Performance":
            # Degradación significativa de accuracy
            if base_acc is not None and acc < base_acc - 0.05:
                msg = f"{name} significantly reduced accuracy (from {base_acc:.2f} to {acc:.2f})."
                conclusions.append(msg)
                test_entry['conclusions'].append(msg)

            # F1-score completamente colapsado
            if f1 == 0.0:
                msg = f"Model completely failed to identify any positive (risky) cases under {name}. F1-score dropped to 0."
                conclusions.append(msg)
                test_entry['conclusions'].append(msg)
                rec = f"Investigate decision boundary under {name}; model may need threshold adjustment or regularization to improve minority class detection."
                recommendations.append(rec)
                test_entry['recommendations'].append(rec)

            # Accuracy crítico
            if acc < 0.75:
                rec = f"Accuracy under {name} is critically low ({acc:.2f}). Consider model retraining or improving data quality."
                recommendations.append(rec)
                test_entry['recommendations'].append(rec)

        tests_summary.append(test_entry)
        return acc, f1


    ensure_dir(os.path.join(FIGURES_DIR_2, model_name))

    # Baseline
    base_acc, base_f1 = test_block(
        "Baseline Performance",
        X, y,
        "Baseline test on original, clean data to benchmark model performance.",
        "No attack applied. This test establishes the model's expected accuracy under normal, unperturbed conditions.",
        base_acc=None
    )

    # FGSM
    try:
        X_fgsm = apply_fgsm(clf, X)
        acc, f1 = test_block(
            "FGSM Attack",
            X_fgsm, y,
            "FGSM introduces adversarial perturbations along the gradient of the loss function.",
            "Fast Gradient Sign Method (ε=0.1) perturbs inputs in the direction of the gradient to cause misclassification with minimal modification.",
            base_acc=base_acc
        )
        if acc < base_acc - 0.1:
            rec = "Implement adversarial training strategies to improve defense against gradient-based attacks like FGSM."
            recommendations.append(rec)
            tests_summary[-1]['recommendations'].append(rec)
        else:
            conc = "Model maintained robust performance under FGSM, indicating resilience to simple gradient-based attacks."
            conclusions.append(conc)
            tests_summary[-1]['conclusions'].append(conc)
    except Exception:
        msg = "FGSM attack could not be applied. Possibly unsupported by the model."
        conclusions.append(msg)
        tests_summary.append({'name': "FGSM Attack", 'description': "N/A", 'conclusions': [msg], 'recommendations': []})

    # PGD
    try:
        X_pgd = apply_pgd(clf, X)
        acc, f1 = test_block(
            "PGD Attack",
            X_pgd, y,
            "PGD is a stronger iterative adversarial attack based on FGSM.",
            "Projected Gradient Descent (ε=0.1, α=0.01, 10 iterations) applies repeated FGSM steps with projection, testing the model’s resilience to stronger perturbations.",
            base_acc=base_acc
        )
        if acc < base_acc - 0.15:
            rec = "Consider robust optimization or certified defenses to mitigate stronger iterative attacks like PGD."
            recommendations.append(rec)
            tests_summary[-1]['recommendations'].append(rec)
        else:
            conc = "Model showed acceptable performance under PGD, suggesting moderate robustness to iterative adversarial attacks."
            conclusions.append(conc)
            tests_summary[-1]['conclusions'].append(conc)
    except Exception:
        msg = "PGD attack could not be applied. Possibly unsupported by the model."
        conclusions.append(msg)
        tests_summary.append({'name': "PGD Attack", 'description': "N/A", 'conclusions': [msg], 'recommendations': []})

    # Gaussian Noise
    X_noise = add_gaussian_noise(X)
    acc, f1 = test_block(
        "Gaussian Noise Injection",
        X_noise, y,
        "Test with additive Gaussian noise simulating sensor or measurement errors.",
        "Applies Gaussian noise (mean=0, std=0.1) to each feature to evaluate how noise in inputs affects predictions.",
        base_acc=base_acc
    )
    if acc < base_acc - 0.05:
        rec = "Introduce data augmentation techniques or feature smoothing to mitigate noise sensitivity."
        recommendations.append(rec)
        tests_summary[-1]['recommendations'].append(rec)
    else:
        conc = "Model handled Gaussian noise well, indicating good generalization under noisy inputs."
        conclusions.append(conc)
        tests_summary[-1]['conclusions'].append(conc)

    # Boundary Testing
    X_bound = boundary_testing(X)
    acc, f1 = test_block(
        "Boundary Value Testing",
        X_bound, y,
        "Boundary value test forces features to extreme values.",
        "For each feature, inputs are set to either their minimum or maximum observed value, probing edge-case behavior.",
        base_acc=base_acc
    )
    if acc < base_acc - 0.05:
        rec = "Ensure training data includes sufficient coverage of boundary values or apply regularization."
        recommendations.append(rec)
        tests_summary[-1]['recommendations'].append(rec)
    else:
        conc = "Model maintained stability under boundary value conditions, suggesting appropriate handling of extreme inputs."
        conclusions.append(conc)
        tests_summary[-1]['conclusions'].append(conc)

    # Label Flipping
    y_flipped = flip_labels(y)
    acc, f1 = test_block(
        "Label Noise Simulation",
        X, y_flipped,
        "Simulates label corruption by flipping 10% of class labels randomly.",
        "This test does not modify features, only labels. It assesses robustness to errors in training or drift in label quality.",
        base_acc=base_acc
    )
    if f1 < 0.6:
        rec = "Explore robust loss functions (e.g., Huber loss) or label noise correction strategies."
        recommendations.append(rec)
        tests_summary[-1]['recommendations'].append(rec)
    else:
        conc = "Model demonstrated reasonable resistance to moderate label noise."
        conclusions.append(conc)
        tests_summary[-1]['conclusions'].append(conc)

    # Render final HTML
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(['html']))
    template = env.get_template('robustness_report.html.j2')

    rendered = template.render(
        model_name=model_name,
        test_results=tests_summary,
        conclusions=conclusions,
        recommendations=recommendations
    )

    out_path = os.path.join(REPORT_DIR_2, f"{model_name}_robustness_report.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(rendered)
    print(f"Robustness report saved to {out_path}")

