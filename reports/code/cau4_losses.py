"""Cau 4. TAT CA CAC HAM LOSS (hoi quy, phan loai nhi phan/da lop, focal, regularization, cay).

Assert doi chieu voi vi du so trong slide 26-32 (ML-04) va slide 14-15 (ML-06).
Chay: python reports/code/cau4_losses.py
"""
import numpy as np


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


if __name__ == "__main__":
    y, yh = np.array([3.0, 5, 2.5, 7]), np.array([2.5, 5, 4, 8])
    print(f"MSE={mse(y, yh):.4f} RMSE={rmse(y, yh):.4f} MAE={mae(y, yh):.4f} MBE={mbe(y, yh):.4f}")

    s = np.array([2.0, 0.5, -0.5, -2.0])           # y = +1 cho ca 4 mau
    print("y*s           :", s)
    print("Perceptron    :", perceptron_loss(1, s))
    print("Hinge         :", hinge_loss(1, s))
    print("Logistic      :", logistic_loss(1, s).round(4))

    # Vi du hinge loss da lop slide 28 (cot = lop, hang = anh)
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

    # Entropy / Info gain / Gini (slide 14-15 ML-06: Outlook tren weather data)
    e_play = entropy([9, 5])
    e_outlook = 5 / 14 * entropy([3, 2]) + 4 / 14 * entropy([4, 0]) + 5 / 14 * entropy([2, 3])
    print(f"Entropy(Play)={e_play:.3f}  Entropy(Outlook)={e_outlook:.3f}  "
          f"Gain={e_play - e_outlook:.3f}  Gini(sunny)={gini([3, 2]):.2f}")
    assert round(e_play, 2) == 0.94 and abs(e_outlook - 0.693) < 1e-3

    # Regularization L1/L2 (slide 33-36)
    w, lam, base = np.array([0.5, -2.0, 0.0, 3.0]), 0.1, 1.0
    print(f"Loss + L1 = {base + lam * np.abs(w).sum():.3f}   Loss + L2 = {base + lam * (w ** 2).sum():.3f}")
