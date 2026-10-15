import os
import cv2
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

def load_system(models_dir="models"):
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    model_path = os.path.join(models_dir, "best_classifier.pkl")
    labels_path = os.path.join(models_dir, "labels.json")

    if not (os.path.exists(pca_path) and os.path.exists(model_path) and os.path.exists(labels_path)):
        raise FileNotFoundError("Trained models not found. Please run 'train_and_evaluate.py' first.")

    pca = joblib.load(pca_path)
    model = joblib.load(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    return pca, model, labels

def log_attendance(student_id, student_name, confidence, log_file="attendance.csv"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Avoid duplicate check-in within the same day for the same student
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        already_checked = ((df["Date"] == date_str) & (df["Student_ID"] == str(student_id))).any()
        if already_checked:
            return False
    else:
        df = pd.DataFrame(columns=["Date", "Time", "Student_ID", "Student_Name", "Confidence_Score"])

    new_record = pd.DataFrame([{
        "Date": date_str,
        "Time": time_str,
        "Student_ID": str(student_id),
        "Student_Name": student_name,
        "Confidence_Score": f"{confidence:.2%}"
    }])

    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(log_file, index=False, encoding="utf-8-sig")
    print(f"[CHECK-IN] {student_name} ({student_id}) checked in at {time_str} ({confidence:.2%})")
    return True

def run_attendance(confidence_threshold=0.50):
    pca, model, labels = load_system()
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("\n[INFO] Starting Attendance System.")
    print(f"[INFO] Confidence Threshold: {confidence_threshold:.0%}")
    print("[INFO] Press 'q' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (100, 100), interpolation=cv2.INTER_AREA)
            face_norm = cv2.equalizeHist(face_resized).flatten().astype(np.float32) / 255.0

            # Project into PCA feature space
            face_pca = pca.transform([face_norm])

            # Predict identity
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(face_pca)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
            else:
                pred_idx = model.predict(face_pca)[0]
                conf = 1.0

            if conf >= confidence_threshold:
                full_label = labels[pred_idx]
                parts = full_label.split("_", 1)
                student_id = parts[0]
                student_name = parts[1].replace("_", " ") if len(parts) > 1 else full_label

                color = (0, 255, 0)
                display_text = f"{student_name} ({conf:.0%})"
                log_attendance(student_id, student_name, conf)
            else:
                color = (0, 0, 255)
                display_text = f"Unknown ({conf:.0%})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 25), (x + w, y), color, cv2.FILLED)
            cv2.putText(
                frame,
                display_text,
                (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

        cv2.imshow("Student Face Attendance - Machine Learning", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance()
