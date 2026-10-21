"""Cau 5. CAC DO DO HIEU SUAT: ACC, Precision, Recall/Sensitivity, Specificity, F1, ROC, AUC.

Vi du u ac tinh (slide 20, ML-04) + ROC/AUC cua 3 mo hinh.
Chay: python reports/code/cau5_metrics.py [roc.png]   (tham so tuy chon: luu do thi ROC)
"""
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def metrics_from_cm(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)                 # = sensitivity = TPR
    specificity = tn / (tn + fp)            # = TNR = 1 - FPR
    f1 = 2 * precision * recall / (precision + recall)
    return dict(acc=acc, precision=precision, recall=recall, specificity=specificity, f1=f1)


def auc_trapezoid(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# Vi du u ac tinh (slide 20)
m = metrics_from_cm(tp=1, fp=1, tn=90, fn=8)
print({k: round(v, 3) for k, v in m.items()})
assert round(m["acc"], 2) == 0.91 and m["precision"] == 0.5 and round(m["recall"], 2) == 0.11

# Doi chieu voi sklearn
y_true = np.array([1] * 9 + [0] * 91)
y_pred = np.array([1] + [0] * 8 + [1] + [0] * 90)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
assert (tp, fp, tn, fn) == (1, 1, 90, 8)
assert np.isclose(f1_score(y_true, y_pred), m["f1"])
assert np.isclose(precision_score(y_true, y_pred), m["precision"])
assert np.isclose(recall_score(y_true, y_pred), m["recall"])
assert np.isclose(recall_score(y_true, y_pred, pos_label=0), m["specificity"])

# ROC & AUC tren mo hinh that
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.8], random_state=3)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
models = {"LogReg": LogisticRegression(max_iter=1000), "GaussianNB": GaussianNB(),
          "KNN(k=5)": KNeighborsClassifier()}
curves = {}
for name, mdl in models.items():
    score = mdl.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, score)
    auc = roc_auc_score(yte, score)
    assert np.isclose(auc, auc_trapezoid(fpr, tpr))
    curves[name] = (fpr, tpr, auc)
    yh = (score >= 0.5).astype(int)
    print(f"{name:<10} ACC={accuracy_score(yte, yh):.3f} P={precision_score(yte, yh):.3f} "
          f"R={recall_score(yte, yh):.3f} Spec={recall_score(yte, yh, pos_label=0):.3f} "
          f"F1={f1_score(yte, yh):.3f} AUC={auc:.3f}")

if len(sys.argv) > 1:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ngau nhien (AUC=0.5)")
    ax.set_xlabel("FPR = 1 - Specificity")
    ax.set_ylabel("TPR = Recall / Sensitivity")
    ax.set_title("Duong cong ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=150)
