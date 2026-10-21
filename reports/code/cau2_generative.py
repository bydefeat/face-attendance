"""Cau 2. GENERATIVE sinh duoc du lieu moi, DISCRIMINATIVE thi khong.

Vi du Cat & Dog 2x2 trong slide 13-15 (ML-04-CLASSIFICATION).
Chay: python reports/code/cau2_generative.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

rng = np.random.default_rng(0)

X_dog = np.array([[200, 200, 50, 50],
                  [210, 190, 40, 60]], dtype=float)

# Generative: hoc P(X | Dog) ~ N(mu, sigma^2) cho tung pixel
mu, sigma = X_dog.mean(axis=0), X_dog.std(axis=0)
z = np.array([0.8, -1.2, 0.4, 0.2])                # z ~ N(0,1) nhu slide
x_new = mu + sigma * z
print("mu =", mu, " sigma =", sigma)
print("Anh cho moi sinh ra:", x_new.round())       # [209 189 47 56]
assert np.allclose(x_new.round(), [209, 189, 47, 56])

# Discriminative (Logistic Regression): chi co ranh gioi w.x + b
w, b = np.array([-1, -1, 1, 1]), 0
sigmoid = lambda t: 1 / (1 + np.exp(-t))
for x in ([0, 0, 255, 255], [-100, -100, 0, 0]):
    x = np.asarray(x, float)
    print(f"x={x}  w.x+b={w @ x + b:>6.0f}  P(Dog|x)={sigmoid(w @ x + b):.3f}")
# Ca 2 anh vo nghia deu duoc goi la "Dog": w chi cho biet x nam ben nao cua ranh gioi,
# khong cho biet P(x) nen khong co gi de lay mau.

# Voi sklearn: GaussianNB (generative) luu theta_, var_ -> lay mau duoc
X, y = make_classification(n_samples=500, n_features=4, n_informative=4,
                           n_redundant=0, random_state=1)
nb = GaussianNB().fit(X, y)
fake = rng.normal(nb.theta_[1], np.sqrt(nb.var_[1]), size=(5, 4))
print("GaussianNB sinh 5 mau lop 1 ->", nb.predict(fake))
lr = LogisticRegression().fit(X, y)
print("LogReg chi co coef_ =", lr.coef_.round(2), "-> khong co phan bo de lay mau")
