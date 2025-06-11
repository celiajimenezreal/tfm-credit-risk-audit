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
    # Ensure shape & classes
    if getattr(classifier, '_input_shape', None) is None:
        classifier._input_shape = (X.shape[1],)
    if getattr(classifier, '_nb_classes', None) is None:
        try:
            classifier._nb_classes = classifier.predict(X[:1]).shape[1]
        except Exception:
            pass
    try:
        if attack.lower() == 'fgsm':
            atk = FastGradientMethod(estimator=classifier, eps=eps, **kwargs)
        elif attack.lower() == 'pgd':
            atk = ProjectedGradientDescent(estimator=classifier, eps=eps, **kwargs)
        else:
            raise ValueError(f"Unsupported attack: {attack}")
        return atk.generate(x=X)
    except Exception:
        # Decision-based fallback
        atk = HopSkipJump(classifier=classifier, max_iter=10, max_eval=100, init_eval=10)
        return atk.generate(x=X)


def evaluate_on_adversarial(
    art_classifiers: Dict[str, object],
    X_adv: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    results = {}
    for name, cls in art_classifiers.items():
        preds = np.argmax(cls.predict(X_adv), axis=1)
        results[name] = {
            'accuracy': accuracy_score(y_true, preds),
            'f1_score': f1_score(y_true, preds)
        }
    return results


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
    # Prepare XGB sample and baseline storage
    xgb_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    baseline_metrics: Dict[str, Dict[str, float]] = {}

    # Subsample once for each XGB model
    for name, cls in art_clfs.items():
        if isinstance(cls, XGBoostClassifier):
            idx = np.random.choice(len(X_orig), size=min(500, len(X_orig)), replace=False)
            X_sub, y_sub = X_orig[idx], y_true[idx]
            cls._input_shape = (X_sub.shape[1],)
            cls._nb_classes = len(np.unique(y_sub))
            xgb_data[name] = (X_sub, y_sub)

    # Iterate over epsilons
    for eps in epsilons:
        results_summary[eps] = {}
        perturb_summary[eps] = {}
        for name, cls in art_clfs.items():
            # Determine data
            if isinstance(cls, XGBoostClassifier):
                X_use, y_use = xgb_data[name]
                if eps == 0.0:
                    X_adv = X_use.copy()
                else:
                    print(f"Running HopSkipJump for {name}, eps={eps}")
                    atk = HopSkipJump(classifier=cls, max_iter=10, max_eval=100, init_eval=10)
                    X_adv = atk.generate(x=X_use)
            else:
                X_use, y_use = X_orig, y_true
                X_adv = generate_adversarial_examples(cls, X_use, attack='fgsm', eps=eps)

            # Evaluate
            metrics = evaluate_on_adversarial({name: cls}, X_adv, y_use)[name]
            # Save baseline metrics
            if eps == 0.0:
                baseline_metrics[name] = metrics
            # Perturbation analysis
            df_top = analyze_perturbations(X_use, X_adv, feature_names)

            results_summary[eps][name] = metrics
            perturb_summary[eps][name] = df_top

            # Save plots
            perf_path = os.path.join(figures_dir, f'{name}_attack_perf_eps_{eps}.png')
            pert_path = os.path.join(figures_dir, f'{name}_feat_pert_eps_{eps}.png')
            plot_attack_results_single(name, metrics, eps, perf_path)
            plot_feature_perturbations(df_top, pert_path)

    # Generate HTML report
    if fmt == 'html':
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('<html><head><meta charset="utf-8"><title>Robustness Audit</title></head><body>')
            f.write('<h1>Adversarial Robustness Audit</h1>')
            for eps in epsilons:
                f.write(f'<h2>FGSM Attack (eps={eps})</h2>')
                for name in art_clfs.keys():
                    metrics = results_summary[eps][name]
                    df_top = perturb_summary[eps][name]
                    f.write(f'<h3>Model: {name}</h3>')
                    f.write('<p>')
                    f.write(f'<strong>Accuracy:</strong> {metrics["accuracy"]:.3f}<br>')
                    f.write(f'<strong>F1 score:</strong> {metrics["f1_score"]:.3f}')
                    f.write('</p>')
                    perf_img = os.path.join(figures_dir, f'{name}_attack_perf_eps_{eps}.png')
                    pert_img = os.path.join(figures_dir, f'{name}_feat_pert_eps_{eps}.png')
                    f.write(f'<img src="{perf_img}" alt="Perf"><br>')
                    f.write(f'<img src="{pert_img}" alt="Pert"><br>')
                    f.write(df_top.to_html(index=False))
                    # Add findings for eps > 0
                    if eps > 0.0:
                        base = baseline_metrics[name]
                        delta_acc = base['accuracy'] - metrics['accuracy']
                        delta_f1 = base['f1_score'] - metrics['f1_score']
                        top_feats = df_top['feature'].tolist()[:3]
                        f.write('<h4>Findings</h4>')
                        f.write('<ul>')
                        f.write(f'<li>Accuracy drop vs. baseline: {delta_acc:.2%}.</li>')
                        f.write(f'<li>F1 score drop vs. baseline: {delta_f1:.2%}.</li>')
                        f.write(f'<li>Top sensitive features: {", ".join(top_feats)}.</li>')
                        f.write('</ul>')
                        f.write('<h4>Recommendations</h4>')
                        f.write('<ul>')
                        f.write('<li>Consider adversarial training (PGD) to bolster robustness.</li>')
                        f.write('<li>Regularize or remove high-perturbation features.</li>')
                        f.write('<li>Evaluate ensemble models and robust loss functions.</li>')
                        f.write('</ul>')
                    f.write('<hr>')
            f.write('</body></html>')
    else:
        # Markdown fallback
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# Adversarial Robustness Audit Report\n')
            for eps in epsilons:
                f.write(f'## FGSM Attack (eps={eps})\n')
                for name in art_clfs.keys():
                    metrics = results_summary[eps][name]
                    df_top = perturb_summary[eps][name]
                    f.write(f'### Model: {name}\n')
                    f.write(f'- Accuracy: {metrics["accuracy"]:.3f}\n')
                    f.write(f'- F1 score: {metrics["f1_score"]:.3f}\n')
                    f.write(df_top.to_markdown(index=False) + '\n')
    print(f'Report saved to {report_path}') 