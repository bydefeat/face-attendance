import os
import cv2
import argparse
import time

def capture_student_faces(student_id, student_name, total_samples=50, output_dir="dataset"):
    folder_name = f"{student_id}_{student_name.replace(' ', '_')}"
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print(f"[INFO] Collecting {total_samples} face samples for {student_name} ({student_id}).")
    print("[INFO] Look at the camera and slowly tilt/turn your head slightly.")
    print("[INFO] Press 'q' to cancel anytime.")

    count = 0
    last_capture_time = time.time()

    while count < total_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Samples: {count}/{total_samples}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Capture a frame every 0.15 seconds to ensure pose variation
            if time.time() - last_capture_time > 0.15:
                face_crop = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (100, 100), interpolation=cv2.INTER_AREA)
                
                # Histogram equalization to normalize lighting conditions
                face_normalized = cv2.equalizeHist(face_resized)

                file_path = os.path.join(save_path, f"face_{count:03d}.jpg")
                cv2.imwrite(file_path, face_normalized)

                count += 1
                last_capture_time = time.time()
                break

        cv2.imshow("Capture Face Dataset - OpenCV", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Capture interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Saved {count} samples to '{save_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture student face dataset via webcam")
    parser.add_argument("--id", type=str, required=True, help="Student ID (e.g., 4901104001)")
    parser.add_argument("--name", type=str, required=True, help="Student Full Name (e.g., Nguyen_Van_A)")
    parser.add_argument("--samples", type=int, default=50, help="Number of face images to capture (default: 50)")
    args = parser.parse_args()

    capture_student_faces(args.id, args.name, args.samples)
