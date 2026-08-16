"""
CODSOFT Internship - Task 2
Face Detection & Recognition
"""

import cv2
import os
import argparse
import numpy as np

# PART A: FACE DETECTION
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_faces_in_image(image_path, output_path="detected_output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    print(f"Detected {len(faces)} face(s).")
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to: {output_path}")
    return faces


def detect_faces_webcam():
    """Opens the default webcam and draws boxes around faces in real time.
    Press 'q' to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.imshow("Face Detection - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# PART B (BONUS): FACE RECOGNITION with LBPH
def train_face_recognizer(dataset_dir="dataset", model_path="face_model.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}

    for label_id, person_name in enumerate(sorted(os.listdir(dataset_dir))):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[label_id] = person_name

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            detected = face_cascade.detectMultiScale(img, 1.1, 5)
            for (x, y, w, h) in detected:
                faces.append(img[y:y + h, x:x + w])
                labels.append(label_id)

    if not faces:
        print("No faces found in dataset - check your folder structure.")
        return None

    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)
    print(f"Model trained on {len(faces)} face samples across {len(label_map)} people.")
    print(f"Saved model to {model_path}")
    return label_map


def recognize_face(image_path, model_path="face_model.yml", label_map=None):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label_id, confidence = recognizer.predict(face_roi)
        name = label_map.get(label_id, "Unknown") if label_map else str(label_id)
        text = f"{name} ({confidence:.0f})"
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite("recognized_output.jpg", img)
    print("Saved result to recognized_output.jpg")


# DEMO / CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection & Recognition")
    parser.add_argument("--image", type=str, help="Path to an image file to run detection on")
    parser.add_argument("--webcam", action="store_true", help="Run live detection from webcam")
    parser.add_argument("--train", type=str, help="Path to dataset folder to train recognizer")
    parser.add_argument("--recognize", type=str, help="Path to image to recognize (needs trained model)")
    args = parser.parse_args()

    if args.image:
        detect_faces_in_image(args.image)
    elif args.webcam:
        detect_faces_webcam()
    elif args.train:
        train_face_recognizer(args.train)
    elif args.recognize:
        recognize_face(args.recognize)
    else:
        print("No arguments given. Example usage:")
        print("  python task2_face_detection.py --image sample.jpg")
        print("  python task2_face_detection.py --webcam")
