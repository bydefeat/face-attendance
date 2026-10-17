import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

def open_webcam(camera_idx=0):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(camera_idx)

def load_system(models_dir=None):
    if models_dir is None:
        models_dir = BASE_DIR / "models"
    else:
        models_dir = Path(models_dir)

    pca_path = models_dir / "pca_model.pkl"
    model_path = models_dir / "best_classifier.pkl"
    labels_path = models_dir / "labels.json"
    centroids_path = models_dir / "centroids_data.pkl"

    if not (pca_path.exists() and model_path.exists() and labels_path.exists()):
        raise FileNotFoundError(
            f"Chua tim thay mo hinh huan luyen trong '{models_dir}'. "
            "Vui long chay 'python src/train_and_evaluate.py' truoc."
        )

    pca = joblib.load(str(pca_path))
    model = joblib.load(str(model_path))
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    centroids_data = None
    if centroids_path.exists():
        centroids_data = joblib.load(str(centroids_path))

    return pca, model, labels, centroids_data

def init_checked_today_set(log_file=None):
    """
    Khoi tao danh sach sinh vien da diem danh hom nay tu file CSV vao RAM (O(1) lookup).
    Tranh viec doc dia lien tuc o moi frame gay giat lag camera.
    """
    if log_file is None:
        log_file = BASE_DIR / "attendance.csv"
    else:
        log_file = Path(log_file)

    checked_set = set()
    today_str = datetime.now().strftime("%Y-%m-%d")

    if log_file.exists():
        try:
            df = pd.read_csv(log_file, dtype={"Student_ID": str})
            today_records = df[df["Date"] == today_str]
            checked_set = set(today_records["Student_ID"].astype(str).tolist())
            print(f"[CACHE] Da tai {len(checked_set)} sinh vien da diem danh hom nay ({today_str}) vao RAM.")
        except Exception as e:
            print(f"[WARN] Khong doc duoc file lich su diem danh: {e}")

    return checked_set

def log_attendance(student_id, student_name, confidence, checked_today_set, log_file=None):
    """
    Ghi nhat ky diem danh vao file attendance.csv khi sinh vien duoc xac thuc.
    """
    if log_file is None:
        log_file = BASE_DIR / "attendance.csv"
    else:
        log_file = Path(log_file)

    student_id_str = str(student_id)
    if student_id_str in checked_today_set:
        return False

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    new_row = {
        "Date": date_str,
        "Time": time_str,
        "Student_ID": student_id_str,
        "Student_Name": student_name,
        "Confidence_Score": f"{confidence:.2%}"
    }

    if log_file.exists():
        df = pd.read_csv(log_file, dtype={"Student_ID": str})
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row], columns=["Date", "Time", "Student_ID", "Student_Name", "Confidence_Score"])

    df.to_csv(log_file, index=False, encoding="utf-8-sig")
    checked_today_set.add(student_id_str)
    print(f"\n>>> [DIEM DANH THANH CONG] {student_name} (MSSV: {student_id_str}) luc {time_str} | Do tin cay: {confidence:.2%}")
    return True

def run_attendance(confidence_threshold=0.60, camera_idx=0):
    pca, model, labels, centroids_data = load_system()
    checked_today = init_checked_today_set()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Khong the ket noi camera index {camera_idx}.")
        return

    print("\n" + "="*60)
    print("HE THONG DIEM DANH KHUON MAT - REAL-TIME ATTENDANCE")
    print(f"- Nguong tin cay (Confidence Threshold): {confidence_threshold:.0%}")
    print(f"- Phat hien nguoi la (Open-Set PCA Rejection): {'Bat (Centroids)' if centroids_data else 'Tat'}")
    print("- Nhan 'q' tren ban phim de ket thuc chuong trinh.")
    print("="*60 + "\n")

    # Co che Debounce / Xac thuc on dinh: Can nhan dien on dinh trong N frame lien tiep
    STABILITY_FRAMES_REQUIRED = 6
    stability_track = {}
    banner_message = ""
    banner_expiry = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        detected_students_this_frame = set()

        for (x, y, w, h) in faces:
            y1, y2 = max(0, y), min(frame_h, y + h)
            x1, x2 = max(0, x), min(frame_w, x + w)

            face_crop = gray[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                continue

            face_resized = cv2.resize(face_crop, (100, 100), interpolation=cv2.INTER_AREA)
            face_norm = cv2.equalizeHist(face_resized).flatten().astype(np.float32) / 255.0

            # Chieu vao khong gian vector dac trung Eigenfaces bang PCA (ML-02)
            face_pca = pca.transform([face_norm])

            # Du doan xac suat phan lop
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(face_pca)[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
            else:
                pred_idx = int(model.predict(face_pca)[0])
                conf = 1.0

            # Kiem tra khoang cach toi tam cum (Centroid distance) de tu choi nguoi la (Open-Set Recognition)
            is_stranger = False
            if centroids_data:
                centroids = centroids_data.get("centroids", {})
                thresholds = centroids_data.get("thresholds", {})
                if pred_idx in centroids:
                    c_vec = centroids[pred_idx]
                    dist_to_c = float(np.linalg.norm(face_pca[0] - c_vec))
                    max_allowed_dist = thresholds.get(pred_idx, 999.0)
                    if dist_to_c > max_allowed_dist * 1.2:
                        is_stranger = True

            if (not is_stranger) and (conf >= confidence_threshold) and (pred_idx < len(labels)):
                full_label = labels[pred_idx]
                parts = full_label.split("_", 1)
                student_id = parts[0]
                student_name = parts[1].replace("_", " ") if len(parts) > 1 else full_label

                detected_students_this_frame.add(student_id)
                current_hits = stability_track.get(student_id, 0) + 1
                stability_track[student_id] = current_hits

                is_already_checked = student_id in checked_today

                # Khi nhan dien on dinh du so frame quy dinh
                if current_hits >= STABILITY_FRAMES_REQUIRED and not is_already_checked:
                    just_logged = log_attendance(student_id, student_name, conf, checked_today)
                    if just_logged:
                        banner_message = f"DIEM DANH THANH CONG: {student_name} ({student_id})"
                        banner_expiry = time.time() + 3.5

                # Mau sac va chu thich tren bounding box
                if is_already_checked or current_hits >= STABILITY_FRAMES_REQUIRED:
                    box_color = (0, 255, 0)
                    status_text = f"{student_name} - DA DIEM DANH" if is_already_checked else f"{student_name} ({conf:.0%})"
                else:
                    box_color = (0, 215, 255) # Mau vang dang xac thuc
                    status_text = f"Xac thuc: {current_hits}/{STABILITY_FRAMES_REQUIRED} ({conf:.0%})"
            else:
                box_color = (0, 0, 255) # Mau do: Nguoi la hoac khong ro
                status_text = f"Nguoi la (Unknown - {conf:.0%})"

            # Ve bounding box va nhan thong tin
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x2, y1), box_color, cv2.FILLED)
            cv2.putText(
                frame,
                status_text,
                (x1 + 6, max(15, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

        # Giam counter cua cac hoc sinh khong con trong khung hinh (Debounce decay)
        for sid in list(stability_track.keys()):
            if sid not in detected_students_this_frame:
                stability_track[sid] = max(0, stability_track[sid] - 1)
                if stability_track[sid] == 0:
                    del stability_track[sid]

        # Hien thi Banner thong bao khi diem danh thanh cong
        if time.time() < banner_expiry and banner_message:
            cv2.rectangle(frame, (0, 0), (frame_w, 45), (0, 180, 0), cv2.FILLED)
            cv2.putText(
                frame,
                banner_message,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imshow("He thong Diem danh Sinh vien - Machine Learning (Nhan 'q' de thoat)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Da dong camera va ket thuc chuong trinh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chay ung dung diem danh sinh vien bang khuon mat (Real-time)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Nguong do tin cay nhan dien (0.0 - 1.0, mac dinh 0.60)")
    parser.add_argument("--camera", type=int, default=0, help="Index webcam (Mac dinh 0)")
    args = parser.parse_args()

    run_attendance(confidence_threshold=args.threshold, camera_idx=args.camera)
