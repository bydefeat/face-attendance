"""Code minh hoa cho 5 cau hoi on tap ML (dua tren bai giang ML-01, 03, 04, 06, 07).

Chay:  python reports/on_tap_ml.py
Moi phan co assert doi chieu voi vi du so trong slide -> sai cong thuc la bao loi ngay.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)


# =====================================================================
# CAU 1. PARAMETRIC vs NON-PARAMETRIC
# =====================================================================
def cau1_parametric_vs_nonparametric():
    print("\n=== CAU 1: Parametric vs Non-parametric ===")
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


# =====================================================================
# CAU 2. GENERATIVE vs DISCRIMINATIVE (vi du Cat & Dog 2x2 trong slide 13-15)
# =====================================================================
def cau2_generative_vs_discriminative():
    print("\n=== CAU 2: Generative sinh du lieu, Discriminative thi khong ===")
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


# =====================================================================
# CAU 3. LOGISTIC REGRESSION: hoi quy tren log-odds, dung de phan loai
# =====================================================================
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def logistic_newton_raphson(X, y, n_iter=25):
    """Uoc luong beta bang Newton-Raphson / IRLS (slide 24-26)."""
    X1 = np.c_[np.ones(len(X)), X]
    beta = np.zeros(X1.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X1 @ beta)
        W = p * (1 - p)
        grad = X1.T @ (y - p)                    # score vector
        H = -(X1.T * W) @ X1                     # Hessian
        beta = beta - np.linalg.solve(H, grad)
    return beta


def cau3_logistic_regression():
    print("\n=== CAU 3: Logistic Regression ===")
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


# =====================================================================
# CAU 4. TAT CA CAC HAM LOSS
# =====================================================================
# --- Hoi quy ---
def mse(y, yh):  return np.mean((y - yh) ** 2)            # L2 / quadratic
def rmse(y, yh): return np.sqrt(mse(y, yh))
def mae(y, yh):  return np.mean(np.abs(y - yh))           # L1
def mbe(y, yh):  return np.mean(y - yh)                   # mean bias error (co dau)


# --- Phan loai nhi phan, y in {-1, +1}, s = diem so cua mo hinh ---
def perceptron_loss(y, s): return np.maximum(0, -y * s)
def hinge_loss(y, s):      return np.maximum(0, 1 - y * s)
def logistic_loss(y, s):   return np.log2(1 + np.exp(-y * s))


# --- Phan loai da lop ---
def multiclass_hinge(scores, y, squared=False):
    """scores: (n, C), y: chi so lop dung. L_i = sum_{j!=y} max(0, s_j - s_y + 1)."""
    s_y = scores[np.arange(len(y)), y][:, None]
    margins = np.maximum(0, scores - s_y + 1)
    margins[np.arange(len(y)), y] = 0
    if squared:
        margins = margins ** 2
    return margins.sum(axis=1)


def softmax(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(scores, y):
    return -np.log(softmax(scores)[np.arange(len(y)), y])


def binary_cross_entropy(y, p):          # y in {0,1}
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def focal_loss(p_t, gamma=2.0, alpha_t=1.0):
    return -alpha_t * (1 - p_t) ** gamma * np.log(p_t)


# --- Do do tap chat cua cay quyet dinh ---
def entropy(counts):
    p = np.asarray(counts, float) / np.sum(counts)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def gini(counts):
    p = np.asarray(counts, float) / np.sum(counts)
    return 1 - (p ** 2).sum()


def cau4_losses():
    print("\n=== CAU 4: Cac ham Loss ===")
    y, yh = np.array([3.0, 5, 2.5, 7]), np.array([2.5, 5, 4, 8])
    print(f"MSE={mse(y, yh):.4f} RMSE={rmse(y, yh):.4f} MAE={mae(y, yh):.4f} MBE={mbe(y, yh):.4f}")

    s = np.array([2.0, 0.5, -0.5, -2.0])           # y = +1 cho ca 4 mau
    print("y*s           :", s)
    print("Perceptron    :", perceptron_loss(1, s))
    print("Hinge         :", hinge_loss(1, s))
    print("Logistic      :", logistic_loss(1, s).round(4))

    # Vi du hinge loss da lop slide 28 (cot = anh, hang = lop cat/car/frog)
    scores = np.array([[-0.39, 1.49, 4.21],        # anh 1, lop dung o cot 0
                       [-4.61, 3.28, 1.46],        # anh 2, lop dung o cot 1
                       [1.03, -2.37, -2.27]])      # anh 3, lop dung o cot 2
    y_true = np.array([0, 1, 2])
    L = multiclass_hinge(scores, y_true)
    print("Multiclass hinge:", L.round(2), " mean =", L.mean().round(2))
    assert np.allclose(L, [8.48, 0.0, 5.2], atol=0.01)
    print("Squared hinge   :", multiclass_hinge(scores, y_true, squared=True).round(2))
    print("Cross-entropy   :", cross_entropy(scores, y_true).round(4))
    print("Binary CE p=0.9,y=1:", binary_cross_entropy(1, 0.9).round(4))

    # Focal loss (slide 32)
    for q in (0.5, 0.9, 0.1):
        print(f"q_dog={q}: CE={-np.log(q):.4f}  FL(gamma=2)={focal_loss(q):.5f}")
    assert round(focal_loss(0.5), 3) == 0.173 and round(focal_loss(0.9), 5) == 0.00105
    assert np.isclose(focal_loss(0.5, gamma=0), -np.log(0.5))   # gamma=0 -> CE

    # Entropy / Info gain / Gini (slide 14-15: Outlook tren weather data)
    e_play = entropy([9, 5])
    e_outlook = 5 / 14 * entropy([3, 2]) + 4 / 14 * entropy([4, 0]) + 5 / 14 * entropy([2, 3])
    print(f"Entropy(Play)={e_play:.3f}  Entropy(Outlook)={e_outlook:.3f}  "
          f"Gain={e_play - e_outlook:.3f}  Gini(sunny)={gini([3, 2]):.2f}")
    assert round(e_play, 2) == 0.94 and abs(e_outlook - 0.693) < 1e-3

    # Regularization L1/L2 (slide 33-36)
    w, lam, base = np.array([0.5, -2.0, 0.0, 3.0]), 0.1, 1.0
    print(f"Loss + L1 = {base + lam * np.abs(w).sum():.3f}   Loss + L2 = {base + lam * (w ** 2).sum():.3f}")


# =====================================================================
# CAU 5. CAC DO DO HIEU SUAT
# =====================================================================
def metrics_from_cm(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)                 # = sensitivity = TPR
    specificity = tn / (tn + fp)            # = TNR = 1 - FPR
    f1 = 2 * precision * recall / (precision + recall)
    return dict(acc=acc, precision=precision, recall=recall, specificity=specificity, f1=f1)


def auc_trapezoid(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def cau5_metrics(save_plot=None):
    print("\n=== CAU 5: Do do hieu suat ===")
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

    if save_plot:
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
        fig.savefig(save_plot, dpi=150)


if __name__ == "__main__":
    import sys
    cau1_parametric_vs_nonparametric()
    cau2_generative_vs_discriminative()
    cau3_logistic_regression()
    cau4_losses()
    cau5_metrics(save_plot=sys.argv[1] if len(sys.argv) > 1 else None)
    print("\nTat ca kiem tra deu PASS.")
