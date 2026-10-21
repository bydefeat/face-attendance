# Câu 4: cài lại các hàm loss trong bài giảng bằng numpy
import numpy as np


# ---------- hồi quy ----------
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def mbe(y, y_hat):
    return np.mean(y - y_hat)


def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))


# ---------- phân loại nhị phân: y là -1 hoặc +1, s là điểm mô hình chấm ----------
def perceptron_loss(y, s):
    return max(0, -y * s)


def hinge_loss(y, s):
    return max(0, 1 - y * s)


def logistic_loss(y, s):
    return np.log2(1 + np.exp(-y * s))


def binary_cross_entropy(y, p):
    # bản dùng xác suất, y là 0 hoặc 1
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------- nhiều lớp ----------
def multiclass_hinge(scores, dung, binh_phuong=False):
    # scores: điểm của một mẫu cho từng lớp, dung: chỉ số lớp đúng
    loss = 0
    for j in range(len(scores)):
        if j == dung:
            continue
        l = max(0, scores[j] - scores[dung] + 1)
        loss += l ** 2 if binh_phuong else l
    return loss


def cross_entropy(scores, dung):
    # softmax rồi lấy -log xác suất của lớp đúng
    p = np.exp(scores) / np.sum(np.exp(scores))
    return -np.log(p[dung])


def focal_loss(p_t, gamma=2, alpha=1):
    return -alpha * (1 - p_t) ** gamma * np.log(p_t)


# ---------- độ đo tạp chất cho cây quyết định ----------
def entropy(dem):
    p = np.array(dem) / sum(dem)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def gini(dem):
    p = np.array(dem) / sum(dem)
    return 1 - np.sum(p ** 2)


print("== Hồi quy ==")
y = np.array([3, 5, 2.5, 7])
y_hat = np.array([2.5, 5, 4, 8])
print("MSE =", mse(y, y_hat), "| MAE =", mae(y, y_hat), "| MBE =", mbe(y, y_hat),
      "| RMSE =", round(rmse(y, y_hat), 4))

print("\n== Nhị phân, nhãn thật y = +1 ==")
for s in [2, 0.5, -0.5, -2]:
    print(f"s = {s:>4}: perceptron = {perceptron_loss(1, s)}, hinge = {hinge_loss(1, s)}, "
          f"logistic = {logistic_loss(1, s):.4f}")
print("BCE khi y = 1, p = 0.9:", round(binary_cross_entropy(1, 0.9), 4))

print("\n== Nhiều lớp, ví dụ slide 28 ==")
# mỗi dòng là một ảnh, mỗi cột là điểm của một lớp; lớp đúng lần lượt là 0, 1, 2
scores = np.array([[-0.39, 1.49, 4.21],
                   [-4.61, 3.28, 1.46],
                   [1.03, -2.37, -2.27]])
lop_dung = [0, 1, 2]
for i in range(3):
    print(f"ảnh {i + 1}: hinge = {multiclass_hinge(scores[i], lop_dung[i]):.2f}, "
          f"squared hinge = {multiclass_hinge(scores[i], lop_dung[i], True):.2f}, "
          f"cross-entropy = {cross_entropy(scores[i], lop_dung[i]):.4f}")
# slide: 8.48, 0, 5.2

print("\n== Focal loss với gamma = 2, ví dụ slide 32 ==")
for q in [0.5, 0.9, 0.1]:
    print(f"q = {q}: CE = {-np.log(q):.4f}, focal = {focal_loss(q):.5f}")
print("gamma = 0 thì focal loss bằng CE:", np.isclose(focal_loss(0.5, gamma=0), -np.log(0.5)))

print("\n== Cây quyết định, thuộc tính Outlook (slide 14-16, ML-06) ==")
e_play = entropy([9, 5])
# sunny 3-2, overcast 4-0, rainy 2-3
e_outlook = 5 / 14 * entropy([3, 2]) + 4 / 14 * entropy([4, 0]) + 5 / 14 * entropy([2, 3])
gain = e_play - e_outlook
print("Entropy(Play) =", round(e_play, 3))
print("Entropy(Outlook) =", round(e_outlook, 3))
print("Information gain =", round(gain, 3))
print("Gain ratio =", round(gain / entropy([5, 4, 5]), 3))
print("Gini(sunny) =", round(gini([3, 2]), 2))

print("\n== Regularization ==")
w = np.array([0.5, -2, 0, 3])
loss_goc = 1.0
lam = 0.1
print("L1 (Lasso):", round(loss_goc + lam * np.sum(np.abs(w)), 3))
print("L2 (Ridge):", round(loss_goc + lam * np.sum(w ** 2), 3))
