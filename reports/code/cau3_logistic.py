# Câu 3: logistic regression, phần "hồi quy" và phần "phân loại"
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_newton_raphson(X, y, so_vong=25):
    # thêm cột toàn số 1 cho beta0
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(so_vong):
        p = sigmoid(X @ beta)
        gradient = X.T @ (y - p)
        W = np.diag(p * (1 - p))
        hessian = -X.T @ W @ X
        beta = beta - np.linalg.inv(hessian) @ gradient
    return beta


X, y = make_classification(n_samples=400, n_features=2, n_redundant=0, random_state=2)

beta = train_newton_raphson(X, y)
# để C thật lớn cho sklearn gần như không regularize, so sánh mới công bằng
model = LogisticRegression(C=1e10, max_iter=1000).fit(X, y)
print("beta tự tính:", beta.round(3))
print("beta sklearn:", np.round([model.intercept_[0], *model.coef_[0]], 3))

# phần hồi quy: log-odds là hàm tuyến tính của x
p = model.predict_proba(X)[:, 1]
log_odds = np.log(p / (1 - p))
tuyen_tinh = model.intercept_[0] + X @ model.coef_[0]
print("log(p/(1-p)) có bằng b0 + b.x không?", np.allclose(log_odds, tuyen_tinh))

# phần phân loại: cắt ngưỡng 0.5
y_du_doan = (p >= 0.5).astype(int)
print("cắt ngưỡng 0.5 có giống model.predict() không?", (y_du_doan == model.predict(X)).all())
print("accuracy:", (y_du_doan == y).mean())

# odds ratio, ví dụ hút thuốc với ung thư trong slide
odds_ung_thu = 647 / 2
odds_khong_ung_thu = 622 / 27
print("odds ratio:", round(odds_ung_thu / odds_khong_ung_thu, 1))   # slide làm tròn ra 14.1
