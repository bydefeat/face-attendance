import os
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from extract_features import prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent

def get_candidate_models():
    """
    Danh sach 9 mo hinh hoc may theo dung noi dung chuong trinh mon hoc:
    - ML-03: Hoi quy Logistic (Multinomial Logistic Regression)
    - ML-04: Support Vector Machine (Linear SVM, RBF SVM)
    - ML-06: k-Nearest Neighbors (k=3, k=5), Decision Tree (CART), Ensemble (Random Forest, AdaBoost)
    - ML-07: Gaussian Naive Bayes
    """
    return {
        "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, C=10.0, gamma="scale", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

def print_academic_insights(best_model_name, df_results):
    """
    In phan bien luan khoa hoc giai thich ly do vi sao mo hinh dat ket qua nhu tren
    dua theo ly thuyet trong cac bai giang ML-02 den ML-07.
    """
    print("\n" + "="*70)
    print("BIEN LUAN KHOA HOC (ACADEMIC INSIGHTS DUA TREN BAI GIANG)")
    print("="*70)
    print("1. Tac dong cua PCA (ML-02):")
    print("   - Khong gian anh 10,000 chieu duoc chieu xuong cac thanh phan truc giao Eigenfaces,")
    print("     loai bo phan lon da cong tuyen (multicollinearity) va nhieu anh sang.")
    print("2. Vi sao SVM (RBF/Linear) va Logistic Regression thuong dat F1 cao nhat (ML-03 & ML-04):")
    print("   - Khi chieu len Eigenfaces, cac khuon mat cung nguoi tao thanh cum loi trong khong gian Euclid.")
    print("   - SVM voi Max-Margin va Logistic Regression toi uu hoa mat phang phan chia toan cuc rat tot.")
    print("3. Vi sao Decision Tree va Gaussian Naive Bayes thuong kem hon (ML-06 & ML-07):")
    print("   - Decision Tree chi chia theo tung truc truc giao don le (orthogonal splits), kem voi du lieu dac trung ket hop.")
    print("   - Naive Bayes gia dinh cac dac trung doc lap co dieu kien, trong khi cac thanh phan PCA van co quan he phi tuyen phuc tap.")
    print("="*70 + "\n")

def evaluate_models(X_train, X_test, y_train, y_test, label_names, output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR / "reports"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = get_candidate_models()
    results = []

    best_model_name = None
    best_f1 = -1.0
    best_model = None
    best_predictions = None

    print("\n" + "="*70)
    print("BENCHMARKING 9 MACHINE LEARNING CLASSIFIERS (ML-03 -> ML-07)")
    print("="*70)

    for name, clf in models.items():
        # Do thoi gian huan luyen (Train Latency)
        t_start_train = time.time()
        clf.fit(X_train, y_train)
        train_time_ms = (time.time() - t_start_train) * 1000.0

        # Do thoi gian suy luan tren 1 mau (Inference Latency per sample)
        t_start_pred = time.time()
        y_pred = clf.predict(X_test)
        pred_time_ms = (time.time() - t_start_pred) * 1000.0 / len(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Train Time (ms)": round(train_time_ms, 2),
            "Inference Time/Sample (ms)": round(pred_time_ms, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = clf
            best_predictions = y_pred

    df_results = pd.DataFrame(results)
    df_results.sort_values(by="F1-Score", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    print("\n" + df_results.to_string(index=False))

    # Luu ket qua benchmark ra file CSV
    csv_path = output_dir / "benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Bang benchmark duoc luu tai '{csv_path}'.")

    # In bao cao phan loai chi tiet cua mo hinh tot nhat
    print(f"\n[BEST MODEL] '{best_model_name}' dat F1-Score cao nhat: {best_f1:.4f}")
    print("\nClassification Report (ML-04 Metric Analysis):")
    print(classification_report(y_test, best_predictions, target_names=label_names, zero_division=0))

    # Ve va luu ma tran nham lan (Confusion Matrix)
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - Best Model: {best_model_name} (ML-04)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Bieu do ma tran nham lan duoc luu tai '{cm_path}'.")

    print_academic_insights(best_model_name, df_results)

    return best_model_name, best_model, df_results

def run_training():
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    pca_model_path = models_dir / "pca_model.pkl"
    best_model_path = models_dir / "best_classifier.pkl"
    labels_path = models_dir / "labels.json"
    centroids_path = models_dir / "centroids_data.pkl"

    X_train_pca, X_test_pca, y_train, y_test, label_names, pca_obj, centroids, thresholds = prepare_data(
        dataset_dir=BASE_DIR / "dataset",
        test_size=0.25,
        pca_components=0.95,
        pca_save_path=pca_model_path
    )

    best_model_name, best_model, _ = evaluate_models(
        X_train_pca, X_test_pca, y_train, y_test, label_names, output_dir=BASE_DIR / "reports"
    )

    # Luu best classifier
    joblib.dump(best_model, str(best_model_path))
    print(f"[INFO] Mo hinh tot nhat '{best_model_name}' da duoc luu tai '{best_model_path}'.")

    # Luu thong tin nhan lop
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Danh sach nhan lop da duoc luu tai '{labels_path}'.")

    # Luu toa do centroids va nguong khoang cach de phat hien nguoi la
    centroid_payload = {
        "centroids": centroids,
        "thresholds": thresholds
    }
    joblib.dump(centroid_payload, str(centroids_path))
    print(f"[INFO] Du lieu Centroids & Thresholds de phat hien nguoi la da duoc luu tai '{centroids_path}'.")

if __name__ == "__main__":
    run_training()
