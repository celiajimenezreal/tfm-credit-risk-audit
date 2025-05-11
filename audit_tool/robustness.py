# audit_tool/robustness.py


from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np
from sklearn.metrics import accuracy_score

def test_adversarial_attack(model, X, y, attack_type='fgsm'):
    """
    Apply an adversarial attack to the model and measure performance degradation.
    attack_type: Currently only 'fgsm' supported.
    """
    classifier = SklearnClassifier(model=model)

    if attack_type == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, eps=0.1)
    else:
        raise ValueError("Currently only 'fgsm' is supported.")
    
    # Generate adversarial examples
    X_adv = attack.generate(x=X.values)
    
    # Predictions
    y_pred = classifier.predict(X.values).argmax(axis=1)
    y_pred_adv = classifier.predict(X_adv).argmax(axis=1)

    # Accuracy
    acc_clean = accuracy_score(y, y_pred)
    acc_adv = accuracy_score(y, y_pred_adv)

    print(f"Accuracy on clean data: {acc_clean:.4f}")
    print(f"Accuracy on adversarial data: {acc_adv:.4f}")
    print(f"Performance drop due to attack: {acc_clean - acc_adv:.4f}")
