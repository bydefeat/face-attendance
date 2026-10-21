# Câu 2: vì sao mô hình generative sinh được dữ liệu mới còn discriminative thì không
# Dùng lại ví dụ ảnh chó 2x2 trong slide: 4 pixel, nửa trên sáng, nửa dưới tối
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

anh_cho = np.array([[200, 200, 50, 50],
                    [210, 190, 40, 60]])

# ----- generative -----
# coi mỗi pixel có phân phối chuẩn, chỉ cần học trung bình và độ lệch chuẩn
mu = anh_cho.mean(axis=0)
sigma = anh_cho.std(axis=0)
print("mu    =", mu)
print("sigma =", sigma)

# lấy z giống slide cho ra đúng số, muốn ảnh khác thì thay bằng np.random.randn(4)
z = np.array([0.8, -1.2, 0.4, 0.2])
anh_moi = mu + sigma * z
print("ảnh chó mới sinh ra:", anh_moi.round())   # slide: 209 189 47 56

# ----- discriminative (logistic regression) -----
# giả sử train xong được w = [-1, -1, 1, 1], b = 0 như slide
w = np.array([-1, -1, 1, 1])
b = 0


def p_dog(x):
    return 1 / (1 + np.exp(-(np.dot(w, x) + b)))


# mô hình chỉ biết "w.x + b > 0 là chó", nên ảnh vô lý cỡ nào cũng được tính là chó
# miễn x3 + x4 > x1 + x2
for x in [[0, 0, 255, 255], [-100, -100, 0, 0]]:
    print(x, "-> P(chó) =", round(p_dog(x), 3))

# ----- thử với sklearn -----
# GaussianNB lưu sẵn trung bình (theta_) và phương sai (var_) của từng lớp,
# lấy mẫu từ đó là có dữ liệu mới. LogisticRegression chỉ có coef_ nên không làm được.
X, y = make_classification(n_samples=500, n_features=4, n_informative=4,
                           n_redundant=0, random_state=1)
nb = GaussianNB().fit(X, y)
mau_moi = np.random.default_rng(0).normal(nb.theta_[1], np.sqrt(nb.var_[1]), size=(5, 4))
print("5 mẫu sinh từ lớp 1, model đoán là:", nb.predict(mau_moi))
