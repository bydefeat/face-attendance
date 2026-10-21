"""Cau 1. PARAMETRIC vs NON-PARAMETRIC.

So tham so cua mo hinh tham so (Logistic Regression) co dinh, con mo hinh phi tham so
(KNN, Decision Tree) tang theo so mau huan luyen.
Chay: python reports/code/cau1_parametric.py
"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

print(f"{'n_train':>8} | {'LogReg (#tham so)':>18} | {'KNN (#so luu)':>14} | {'Tree (#node)':>12}")
for n in [100, 1000, 10000]:
    X, y = make_classification(n_samples=n, n_features=5, random_state=0)
    lr = LogisticRegression().fit(X, y)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    n_lr = lr.coef_.size + lr.intercept_.size     # luon = d + 1, khong doi theo n
    n_knn = knn._fit_X.size                        # KNN "nho" toan bo tap train
    print(f"{n:>8} | {n_lr:>18} | {n_knn:>14} | {tree.tree_.node_count:>12}")
    assert n_lr == 6 and n_knn == n * 5
