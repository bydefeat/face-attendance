# Câu 5: các độ đo đánh giá mô hình phân loại
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ----- ví dụ u ác tính trong slide: 100 người, 9 người u ác tính -----
TP, FP, TN, FN = 1, 1, 90, 8

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)            # còn gọi là sensitivity hay TPR
specificity = TN / (TN + FP)
f1 = 2 * precision * recall / (precision + recall)

print("Tự tính:")
print(f"  accuracy = {accuracy:.2f}, precision = {precision:.2f}, recall = {recall:.2f}, "
      f"specificity = {specificity:.2f}, F1 = {f1:.2f}")

# dựng lại nhãn rồi kiểm tra bằng sklearn
y_that = np.array([1] * 9 + [0] * 91)
y_doan = np.array([1] + [0] * 8 + [1] + [0] * 90)
print("sklearn:")
print("  tn, fp, fn, tp =", confusion_matrix(y_that, y_doan).ravel())
print(f"  accuracy = {accuracy_score(y_that, y_doan):.2f}, "
      f"precision = {precision_score(y_that, y_doan):.2f}, "
      f"recall = {recall_score(y_that, y_doan):.2f}, "
      f"specificity = {recall_score(y_that, y_doan, pos_label=0):.2f}, "
      f"F1 = {f1_score(y_that, y_doan):.2f}")

# ----- ROC và AUC trên dữ liệu lệch lớp (khoảng 80% là lớp 0) -----
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.8], random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

cac_model = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gaussian NB": GaussianNB(),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

plt.figure(figsize=(6, 5))
for ten, model in cac_model.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    diem = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, diem)
    auc = roc_auc_score(y_test, diem)

    print(f"\n{ten}")
    print(f"  acc = {accuracy_score(y_test, y_pred):.3f}, precision = {precision_score(y_test, y_pred):.3f}, "
          f"recall = {recall_score(y_test, y_pred):.3f}, "
          f"specificity = {recall_score(y_test, y_pred, pos_label=0):.3f}, "
          f"F1 = {f1_score(y_test, y_pred):.3f}")
    # AUC đúng nghĩa là diện tích dưới đường ROC, tính tay bằng hình thang cũng ra y vậy
    print(f"  AUC = {auc:.3f} (tính bằng hình thang: {np.trapezoid(tpr, fpr):.3f})")
    plt.plot(fpr, tpr, label=f"{ten} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Đoán bừa (AUC = 0.5)")
plt.xlabel("FPR = 1 - specificity")
plt.ylabel("TPR = recall")
plt.title("Đường ROC")
plt.legend()
plt.savefig("roc.png", dpi=150)
plt.show()
