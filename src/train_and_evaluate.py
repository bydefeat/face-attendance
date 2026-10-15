import os
import json
import time
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from extract_features import prepare_data

def get_candidate_models():
    """
    Returns candidate models aligning with the Machine Learning syllabus.
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

def evaluate_models(X_train, X_test, y_train, y_test, label_names, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    models = get_candidate_models()
    results = []

    best_model_name = None
    best_f1 = -1.0
    best_model = None
    best_predictions = None

    print("\n" + "="*70)
    print("BENCHMARKING MACHINE LEARNING CLASSIFIERS")
    print("="*70)

    for name, clf in models.items():
        # Measure training time
        t_start_train = time.time()
        clf.fit(X_train, y_train)
        train_time_ms = (time.time() - t_start_train) * 1000.0

        # Measure inference time
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

    # Save benchmark table
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved benchmark results to '{csv_path}'.")

    # Plot Confusion Matrix for the best model
    print(f"\n[BEST MODEL] '{best_model_name}' achieved highest F1-Score: {best_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions, target_names=label_names, zero_division=0))

    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - Best Model: {best_model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Confusion Matrix plot to '{cm_path}'.")

    return best_model_name, best_model, df_results

def run_training():
    pca_model_path = "models/pca_model.pkl"
    best_model_path = "models/best_classifier.pkl"
    labels_path = "models/labels.json"

    X_train_pca, X_test_pca, y_train, y_test, label_names, _ = prepare_data(
        dataset_dir="dataset",
        test_size=0.25,
        pca_components=0.95,
        pca_save_path=pca_model_path
    )

    best_model_name, best_model, _ = evaluate_models(
        X_train_pca, X_test_pca, y_train, y_test, label_names
    )

    # Save best model
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    joblib.dump(best_model, best_model_path)
    print(f"[INFO] Best model '{best_model_name}' saved to '{best_model_path}'.")

    # Save label mapping
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Label mapping saved to '{labels_path}'.")

if __name__ == "__main__":
    run_training()
