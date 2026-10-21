# Câu 1: so sánh mô hình tham số và phi tham số
# Tăng dần số mẫu train rồi xem mỗi mô hình phải giữ lại bao nhiêu thứ
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

for n in [100, 1000, 10000]:
    X, y = make_classification(n_samples=n, n_features=5, random_state=0)

    logreg = LogisticRegression().fit(X, y)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)

    # logistic chỉ có 5 hệ số + 1 bias, dữ liệu nhiều hay ít cũng vậy
    so_tham_so = logreg.coef_.size + logreg.intercept_.size

    print(f"n = {n}")
    print("  Logistic Regression:", so_tham_so, "tham số")
    # knn không học gì, nó giữ nguyên tập train để lúc dự đoán tính khoảng cách
    print("  KNN giữ lại:", knn.n_samples_fit_, "mẫu")
    print("  Cây quyết định:", tree.tree_.node_count, "nút")
