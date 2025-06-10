'''
robustness.py

Adversarial Robustness Auditing Module with Per-Model Attack Evaluation and HTML/MD Report Generation

This module includes robust handling for gradient-based and decision-based attacks, ensuring compatibility with all models.
'''

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, HopSkipJump
from art.estimators.classification import SklearnClassifier, XGBoostClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_data(X_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df_X = pd.read_csv(X_path)
    X_test = df_X.values
    feature_names = df_X.columns.tolist()
    y_test = pd.read_csv(y_path)['TARGET'].values
    return X_test, y_test, feature_names


def load_models(model_paths: Dict[str, str]) -> Dict[str, object]:
    import joblib
    return {name: joblib.load(path) for name, path in model_paths.items()}


def wrap_art_classifiers(models: Dict[str, object]) -> Dict[str, object]:
    wrapped = {}
    for name, mdl in models.items():
        if name in ['logreg', 'rf']:
            wrapped[name] = SklearnClassifier(model=mdl)
        elif name == 'xgb':
            wrapped[name] = XGBoostClassifier(model=mdl)
        else:
            raise ValueError(f"Unsupported model: {name}")
    return wrapped


def generate_adversarial_examples(
    classifier,
    X: np.ndarray,
    attack: str = 'fgsm',
    eps: float = 0.1,
    **kwargs
) -> np.ndarray:
    """
    Generate adversarial examples with FGSM/PGD; fallback to HopSkipJump if gradients unavailable.
    Ensures classifier.input_shape and classifier.nb_classes are properly set.
    """
    # Ensure necessary attributes for gradient attacks
    if getattr(classifier, 'input_shape', None) is None:
        classifier.input_shape = (X.shape[1],)
    if not hasattr(classifier, 'nb_classes') or classifier.nb_classes is None:
        try:
            preds = classifier.predict(X[:1])
            classifier.nb_classes = preds.shape[1]
        except Exception:
            pass

    # Try gradient-based attack
    try:
        if attack.lower() == 'fgsm':
            atk = FastGradientMethod(estimator=classifier, eps=eps, **kwargs)
        elif attack.lower() == 'pgd':
            atk = ProjectedGradientDescent(estimator=classifier, eps=eps, **kwargs)
        else:
            raise ValueError(f"Unsupported attack: {attack}")
        return atk.generate(x=X)
    except Exception as e:
        # Fallback to decision-based HopSkipJump
        print(f"Gradient attack failed ({e}); using HopSkipJump for {type(classifier).__name__}.")
        atk = HopSkipJump(classifier=classifier, max_iter=10, max_eval=100, init_eval=10)
        return atk.generate(x=X)


def evaluate_on_adversarial(
    art_classifiers: Dict[str, object],
    X_adv: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    summary = {}
    for name, cls in art_classifiers.items():
        preds = np.argmax(cls.predict(X_adv), axis=1)
        summary[name] = {
            'accuracy': accuracy_score(y_true, preds),
            'f1_score': f1_score(y_true, preds)
        }
    return summary


def analyze_perturbations(
    X_orig: np.ndarray,
    X_adv: np.ndarray,
    feature_names: List[str],
    top_k: int = 10
) -> pd.DataFrame:
    delta = np.abs(X_adv - X_orig)
    mean_delta = delta.mean(axis=0)
    df = pd.DataFrame({'feature': feature_names, 'mean_perturbation': mean_delta})
    df = df.sort_values('mean_perturbation', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df.head(top_k)


def plot_attack_results_single(
    model_name: str,
    metrics: Dict[str, float],
    eps: float,
    save_path: str
) -> None:
    plt.figure()
    plt.bar(['accuracy', 'f1_score'], [metrics['accuracy'], metrics['f1_score']])
    plt.title(f'{model_name} Performance under FGSM (eps={eps})')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_perturbations(
    df_pert: pd.DataFrame,
    save_path: str
) -> None:
    plt.figure()
    plt.barh(df_pert['feature'][::-1], df_pert['mean_perturbation'][::-1])
    plt.xlabel('Mean Perturbation')
    plt.title('Top Feature Perturbations')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_report(
    X_orig: np.ndarray,
    y_true: np.ndarray,
    art_clfs: Dict[str, object],
    epsilons: List[float],
    feature_names: List[str],
    figures_dir: str,
    report_path: str,
    fmt: str = 'html'
) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    results_summary: Dict[float, Dict[str, Dict[str, float]]] = {}
    perturb_summary: Dict[float, Dict[str, pd.DataFrame]] = {}

    for eps in epsilons:
        results_summary[eps] = {}
        perturb_summary[eps] = {}
        for name, cls in art_clfs.items():
            X_adv = generate_adversarial_examples(cls, X_orig, attack='fgsm', eps=eps)
            metrics = evaluate_on_adversarial({name: cls}, X_adv, y_true)[name]
            results_summary[eps][name] = metrics
            df_top = analyze_perturbations(X_orig, X_adv, feature_names)
            perturb_summary[eps][name] = df_top

            perf_path = os.path.join(figures_dir, f'{name}_attack_perf_eps_{eps}.png')
            plot_attack_results_single(name, metrics, eps, perf_path)
            pert_path = os.path.join(figures_dir, f'{name}_feat_pert_eps_{eps}.png')
            plot_feature_perturbations(df_top, pert_path)

    if fmt == 'html':
        with open(report_path, 'w') as f:
            f.write('<html><head><title>Robustness Audit</title></head><body>')
            f.write('<h1>Adversarial Robustness Audit</h1>')
            for eps in epsilons:
                f.write(f'<h2>FGSM Attack (eps={eps})</h2>')
                for name in art_clfs.keys():
                    metrics = results_summary[eps][name]
                    f.write(f'<h3>Model: {name}</h3>')
                    #f.write(f'<p>Accuracy: {metrics["accuracy"]:.3f}, F1: {metrics["f1_score"]:.3f}</p>')
                    perf_img = os.path.join(figures_dir, f'{name}_attack_perf_eps_{eps}.png')
                    pert_img = os.path.join(figures_dir, f'{name}_feat_pert_eps_{eps}.png')
                    f.write(f'<img src="{perf_img}" alt="Perf {name} eps={eps}"><br>')
                    f.write(f'<img src="{pert_img}" alt="Pert {name} eps={eps}"><br>')
                    f.write(perturb_summary[eps][name].to_html(index=False))
            f.write('</body></html>')
    else:
        with open(report_path, 'w') as f:
            f.write('# Adversarial Robustness Audit Report\n')
            for eps in epsilons:
                f.write(f'## FGSM Attack (eps={eps})\n')
                for name in art_clfs.keys():
                    metrics = results_summary[eps][name]
                    f.write(f'### Model: {name}\n')
                    #f.write(f'- Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}\n')
                    perf_img = os.path.join(figures_dir, f'{name}_attack_perf_eps_{eps}.png')
                    pert_img = os.path.join(figures_dir, f'{name}_feat_pert_eps_{eps}.png')
                    f.write(f'![Perf]({perf_img})\n')
                    f.write(f'![Pert]({pert_img})\n')
                    f.write(perturb_summary[eps][name].to_markdown(index=False) + '\n')
    print(f'Report saved to {report_path}')
