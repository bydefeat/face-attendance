import os
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent

def load_face_dataset(dataset_dir=None, target_size=(100, 100)):
    if dataset_dir is None:
        dataset_dir = BASE_DIR / "dataset"
    else:
        dataset_dir = Path(dataset_dir)

    images = []
    labels = []
    label_names = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Thu muc dataset '{dataset_dir}' khong ton tai.")

    class_dirs = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"Khong tim thay thu muc sinh vien nao trong '{dataset_dir}'. Vui long thu thap du lieu truoc.")
    if len(class_dirs) < 2:
        raise ValueError(
            f"Hien tai chi co {len(class_dirs)} sinh vien ('{class_dirs[0]}'). "
            "Cac thuat toan phan loai (ML-03, ML-04) can it nhat 2 sinh vien tro len de thiet lap ranh gioi phan lop."
        )

    for class_idx, folder_name in enumerate(class_dirs):
        label_names.append(folder_name)
        folder_path = dataset_dir / folder_name

        valid_files = [f for f in folder_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(valid_files) < 5:
            raise ValueError(f"Lop '{folder_name}' chi co {len(valid_files)} anh, can it nhat 5 anh tro len de chia train/test.")

        for img_path in valid_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # cv2.resize expects (width, height)
            if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Chuan hoa ve doan [0, 1] theo bai giang ML-02
            flat_vector = img.flatten().astype(np.float32) / 255.0
            images.append(flat_vector)
            labels.append(class_idx)

    X = np.array(images)
    y = np.array(labels)
    return X, y, label_names

def extract_pca_features(X_train, X_test, n_components=0.95, save_model_path=None):
    """
    Trich xuat dac trung Eigenfaces bang Principal Component Analysis (ML-02).
    n_components=0.95 giu lai 95% tong phuong sai tich luy cua tap du lieu.
    """
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"[PCA] Giam so chieu tu {X_train.shape[1]} xuong con {pca.n_components_} thanh phan chinh.")
    print(f"[PCA] Tong ti le phuong sai giai thich duoc: {np.sum(pca.explained_variance_ratio_):.4f}")

    if save_model_path:
        save_path = Path(save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, str(save_path))
        print(f"[PCA] Da luu mo hinh PCA vao '{save_path}'.")

    return X_train_pca, X_test_pca, pca

def compute_class_centroids(X_pca, y):
    """
    Tinh tam cum (Centroid) cua tung sinh vien trong khong gian dac trung PCA (ML-02 & ML-06).
    Dung de do khoang cach Euclidean, phat hien va tu choi nguoi la (Open-Set Recognition).
    """
    classes = np.unique(y)
    centroids = {}
    thresholds = {}

    for c in classes:
        class_samples = X_pca[y == c]
        centroid = np.mean(class_samples, axis=0)
        centroids[int(c)] = centroid

        # Tinh khoang cach tu cac mau toi centroid
        dists = np.linalg.norm(class_samples - centroid, axis=1)
        # Nguong tu choi: mean + 2.5 * std (khoang 99% tin cay Gauss)
        thresholds[int(c)] = float(np.mean(dists) + 2.5 * np.std(dists) + 1e-4)

    return centroids, thresholds

def prepare_data(dataset_dir=None, test_size=0.25, pca_components=0.95, pca_save_path=None):
    if dataset_dir is None:
        dataset_dir = BASE_DIR / "dataset"
    if pca_save_path is None:
        pca_save_path = BASE_DIR / "models" / "pca_model.pkl"

    X, y, label_names = load_face_dataset(dataset_dir)
    print(f"[DATASET] Da tai {len(X)} mau khuon mat cua {len(label_names)} sinh vien.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[SPLIT] Mau huan luyen (Train): {len(X_train)}, Mau kiem thu (Test): {len(X_test)}")

    X_train_pca, X_test_pca, pca = extract_pca_features(
        X_train, X_test, n_components=pca_components, save_model_path=pca_save_path
    )

    centroids, thresholds = compute_class_centroids(X_train_pca, y_train)

    return X_train_pca, X_test_pca, y_train, y_test, label_names, pca, centroids, thresholds
