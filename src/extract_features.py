import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def load_face_dataset(dataset_dir="dataset", target_size=(100, 100)):
    images = []
    labels = []
    label_names = []

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    class_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if len(class_dirs) == 0:
        raise ValueError(f"No student folders found inside '{dataset_dir}'. Please capture data first.")

    for class_idx, folder_name in enumerate(class_dirs):
        label_names.append(folder_name)
        folder_path = os.path.join(dataset_dir, folder_name)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                if img.shape != target_size:
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Flatten 100x100 image into 10000-d vector and normalize to [0, 1]
                flat_vector = img.flatten().astype(np.float32) / 255.0
                images.append(flat_vector)
                labels.append(class_idx)

    X = np.array(images)
    y = np.array(labels)
    return X, y, label_names

def extract_pca_features(X_train, X_test, n_components=0.95, save_model_path=None):
    """
    Eigenfaces extraction via PCA (Dimensionality Reduction).
    n_components: either an integer or float (e.g., 0.95 keeps 95% variance).
    """
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"[PCA] Reduced dimensions from {X_train.shape[1]} down to {pca.n_components_} components.")
    print(f"[PCA] Total explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(pca, save_model_path)
        print(f"[PCA] Model saved to '{save_model_path}'.")

    return X_train_pca, X_test_pca, pca

def prepare_data(dataset_dir="dataset", test_size=0.25, pca_components=0.95, pca_save_path="models/pca_model.pkl"):
    X, y, label_names = load_face_dataset(dataset_dir)
    print(f"[DATASET] Loaded {len(X)} samples across {len(label_names)} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    X_train_pca, X_test_pca, pca = extract_pca_features(
        X_train, X_test, n_components=pca_components, save_model_path=pca_save_path
    )

    return X_train_pca, X_test_pca, y_train, y_test, label_names, pca
