import os
import sys
import time
import argparse
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

def open_webcam(camera_idx=0):
    """
    Mo webcam voi DirectShow tren Windows de khoi dong nhanh (khong bi tre 3-5 giay).
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    cap = cv2.VideoCapture(camera_idx)
    return cap

def capture_student_faces(student_id, student_name, total_samples=50, output_dir=None, camera_idx=0):
    if output_dir is None:
        output_dir = BASE_DIR / "dataset"
    else:
        output_dir = Path(output_dir)

    folder_name = f"{student_id}_{student_name.replace(' ', '_')}"
    save_path = output_dir / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Khong the ket noi den webcam index {camera_idx}.")
        return

    print(f"\n[INFO] Dang thu thap {total_samples} mau khuon mat cho sinh vien: {student_name} ({student_id})")
    print("[HUONG DAN] Nhin thang vao camera, xoay nhe sang trai/phai de lay goc mat phong phu.")
    print("[THOAT] Nhan 'q' de dung bat cu luc nao.\n")

    count = 0
    last_capture_time = time.time()

    while count < total_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Khong doc duoc frame tu camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        display_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        # Ve thanh tien do (Progress bar) tren giao dien
        bar_width = int((count / total_samples) * (frame_w - 40))
        cv2.rectangle(display_frame, (20, 20), (frame_w - 20, 35), (60, 60, 60), cv2.FILLED)
        cv2.rectangle(display_frame, (20, 20), (20 + bar_width, 35), (0, 200, 0), cv2.FILLED)
        cv2.putText(
            display_frame,
            f"Tien do: {count}/{total_samples} anh ({int(count/total_samples*100)}%)",
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        for (x, y, w, h) in faces:
            # Bounding box an toan khong vuot qua bien frame
            y1, y2 = max(0, y), min(frame_h, y + h)
            x1, x2 = max(0, x), min(frame_w, x + w)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"{student_name}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # Chup cach nhau 0.15 giay de da dang tu the mat
            if time.time() - last_capture_time > 0.15:
                face_crop = gray[y1:y2, x1:x2]
                if face_crop.size == 0 or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                    continue

                face_resized = cv2.resize(face_crop, (100, 100), interpolation=cv2.INTER_AREA)
                
                # Can bang histogram anh sang (ML-02 Data Preprocessing)
                face_normalized = cv2.equalizeHist(face_resized)

                file_path = save_path / f"face_{count:03d}.jpg"
                cv2.imwrite(str(file_path), face_normalized)

                count += 1
                last_capture_time = time.time()
                break

        cv2.imshow("Thu thap du lieu khuon mat - ML-02 (Nhan 'q' de thoat)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Qua trinh thu thap da bi huy boi nguoi dung.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[HOAN TAT] Da luu {count}/{total_samples} mau anh vao '{save_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thu thap du lieu khuon mat sinh vien qua webcam (ML-02)")
    parser.add_argument("--id", type=str, required=True, help="Ma so sinh vien (Vi du: 4901104001)")
    parser.add_argument("--name", type=str, required=True, help="Ho va ten sinh vien (Vi du: Nguyen_Van_A)")
    parser.add_argument("--samples", type=int, default=50, help="So luong anh can chup (Mac dinh: 50)")
    parser.add_argument("--camera", type=int, default=0, help="Index webcam (Mac dinh: 0)")
    args = parser.parse_args()

    capture_student_faces(args.id, args.name, args.samples, camera_idx=args.camera)
