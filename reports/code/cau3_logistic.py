"""Cau 3. LOGISTIC REGRESSION: hoi quy tren log-odds, dung de phan loai.

Tu cai Newton-Raphson / IRLS (slide 24-26, ML-03) va doi chieu voi sklearn.
Chay: python reports/code/cau3_logistic.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def logistic_newton_raphson(X, y, n_iter=25):
    """Uoc luong beta bang Newton-Raphson / IRLS."""
    X1 = np.c_[np.ones(len(X)), X]
    beta = np.zeros(X1.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X1 @ beta)
        W = p * (1 - p)
        grad = X1.T @ (y - p)                    # score vector
        H = -(X1.T * W) @ X1                     # Hessian
        beta = beta - np.linalg.solve(H, grad)
    return beta


X, y = make_classification(n_samples=400, n_features=2, n_redundant=0, random_state=2)
beta = logistic_newton_raphson(X, y)
sk = LogisticRegression(C=1e10, max_iter=1000).fit(X, y)   # C lon = gan nhu khong regularize
print("beta Newton-Raphson:", beta.round(4))
print("beta sklearn       :", np.r_[sk.intercept_, sk.coef_[0]].round(4))
assert np.allclose(beta, np.r_[sk.intercept_, sk.coef_[0]], atol=1e-3)

# Phan "hoi quy": log-odds la ham TUYEN TINH cua x
p = sk.predict_proba(X)[:, 1]
log_odds = np.log(p / (1 - p))
print("log(p/(1-p)) == b0 + b.x ?", np.allclose(log_odds, sk.decision_function(X)))
# Phan "phan loai": dat nguong 0.5 len xac suat p
y_hat = (p >= 0.5).astype(int)
assert (y_hat == sk.predict(X)).all()
print("Accuracy:", accuracy_score(y, y_hat))

# Odds ratio vi du hut thuoc / ung thu (slide 16)
odds_ratio = (647 / 2) / (622 / 27)
print(f"Odds ratio smoking = {odds_ratio:.1f}")
assert round(odds_ratio, 1) == 14.0
