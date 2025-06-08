import os
import numpy as np
import pandas as pd
from collections import defaultdict
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    HopSkipJump,
    BoundaryAttack
)
import matplotlib.pyplot as plt

def run_fgsm(model, X, eps):
    """
    Fast Gradient Sign Method (FGSM) attack.
    """
    clf = SklearnClassifier(model=model, clip_values=(X.min(), X.max()))
    attack = FastGradientMethod(estimator=clf, eps=eps)
    return attack.generate(x=X)


def run_pgd(model, X, eps, max_iter=40):
    """
    Projected Gradient Descent (PGD) attack.
    """
    clf = SklearnClassifier(model=model, clip_values=(X.min(), X.max()))
    attack = ProjectedGradientDescent(estimator=clf, eps=eps, max_iter=max_iter)
    return attack.generate(x=X)


def run_hopskipjump(model, X, max_iter=50):
    """
    HopSkipJump decision-based attack.
    """
    clf = SklearnClassifier(model=model, clip_values=(X.min(), X.max()))
    attack = HopSkipJump(classifier=clf, max_iter=max_iter)
    return attack.generate(x=X)


def run_boundary(model, X, max_iter=50):
    """
    Boundary Attack (decision-based).
    """
    clf = SklearnClassifier(model=model, clip_values=(X.min(), X.max()))
    attack = BoundaryAttack(classifier=clf, max_iter=max_iter)
    return attack.generate(x=X)


# Mapping of attack names to functions
_attack_funcs = {
    "fgsm": run_fgsm,
    "pgd": run_pgd,
    "hopskipjump": run_hopskipjump,
    "boundary": run_boundary
}

def run_robustness(models, X, y, attacks, epsilons, output_dir=None):
    """
    Run adversarial attacks on a set of models and return robustness metrics.

    Parameters:
    - models: dict of {model_name: estimator}
    - X: np.array of input features
    - y: np.array of true labels
    - attacks: list of attack names (keys in _attack_funcs)
    - epsilons: list of epsilons for white-box attacks
    - output_dir: if provided, CSV of results is saved here

    Returns:
    - df: pandas.DataFrame with columns [model, attack, eps, success_rate, avg_distortion]
    - adv_examples: dict mapping (model, attack, eps) to list of (X_orig, X_adv, success_mask)
    """
    results = []
    adv_examples = defaultdict(list)

    for m_name, model in models.items():
        for atk in attacks:
            fn = _attack_funcs.get(atk)
            if fn is None:
                continue
            # Determine epsilon list
            eps_list = epsilons if atk in ("fgsm", "pgd") else [None]

            for eps in eps_list:
                print(f"▶ Running {atk} on {m_name} (eps={eps})")
                # Generate adversarial examples
                if eps is not None:
                    X_adv = fn(model, X, eps=eps)
                else:
                    X_adv = fn(model, X)

                # Predict and compute metrics
                y_pred = model.predict(X_adv)
                success_mask = (y_pred != y)
                success_rate = success_mask.mean()
                avg_dist = np.mean(np.linalg.norm(X_adv - X, axis=1))

                results.append({
                    "model": m_name,
                    "attack": atk,
                    "eps": eps,
                    "success_rate": success_rate,
                    "avg_distortion": avg_dist
                })

                # Store examples for feature analysis
                adv_examples[(m_name, atk, eps)].append((X, X_adv, success_mask))

    df = pd.DataFrame(results)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "robustness_results.csv"), index=False)
    return df, adv_examples


def analyze_feature_vulnerability(adv_examples, feature_names, top_k=5):
    """
    Analyze which features are most frequently changed in successful adversarial examples.

    Parameters:
    - adv_examples: dict from run_robustness
    - feature_names: list of feature column names
    - top_k: number of top features to return per attack

    Returns:
    - vuln: dict mapping (model, attack, eps) to list of (feature_name, count)
    """
    vuln = {}
    n_features = len(feature_names)

    for key, examples in adv_examples.items():
        counts = np.zeros(n_features, dtype=int)
        for X_orig, X_adv, mask in examples:
            Xo = X_orig[mask]
            Xa = X_adv[mask]
            # Detect changes per feature
            diffs = np.abs(Xa - Xo) > 1e-8
            counts += diffs.sum(axis=0)

        # Get top_k features
        top_idx = np.argsort(-counts)[:top_k]
        vuln[key] = [(feature_names[i], int(counts[i])) for i in top_idx]

    return vuln

def plot_robustness(df, output_dir=None):
    """
    Plot robustness curves (success rate vs epsilon) for each model and attack.

    Parameters:
    - df: DataFrame from run_robustness
    - output_dir: if provided, saves plots in this directory
    """
    for m in df['model'].unique():
        plt.figure()
        sub = df[df['model'] == m]
        for atk in sub['attack'].unique():
            part = sub[sub['attack'] == atk]
            eps_vals = part['eps'].fillna(0)
            plt.plot(eps_vals, part['success_rate'], marker='o', label=atk)
        plt.title(f"Robustness: {m}")
        plt.xlabel("ε")
        plt.ylabel("Success Rate")
        plt.legend()
        plt.grid(True)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{m}_robustness.png"))
        plt.show()



def generate_markdown_report(df, vuln, output_path):
    """
    Generate a Markdown report summarizing robustness results and vulnerabilities.

    Parameters:
    - df: DataFrame from run_robustness
    - vuln: output from analyze_feature_vulnerability
    - output_path: file path for the .md report
    """
    with open(output_path, "w") as f:
        f.write("# Informe de Robustez Adversarial\n\n")
        f.write("## 1. Resumen de Robustez\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Variables Más Vulnerables\n\n")
        for (model, attack, eps), features in vuln.items():
            title = f"### Modelo: **{model}**, Ataque: **{attack}**, ε = {eps}"
            f.write(f"{title}\n\n")
            f.write("| Variable | Veces modificada |\n")
            f.write("|:---------|-----------------:|\n")
            for var, cnt in features:
                f.write(f"| {var} | {cnt} |\n")
            f.write("\n")

        f.write("## 3. Recomendaciones Generales\n\n")
        f.write("- **Adversarial training**: incluye ejemplos adversariales en el entrenamiento.\n")
        f.write("- **Regularización de gradiente**: penaliza altas sensibilidades.\n")
        f.write("- **Detectores de anomalías**: filtra inputs con cambios atípicos en variables críticas.\n")
