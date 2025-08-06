import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

DATASET_DIR = "dataset"
UNKNOWN_IMAGE_DIR = "unknown_images"
UNKNOWN_VIDEO_DIR = "unknown_videos"
VIDEO_CLIP_SECONDS = 3
# Cooldown between unknown recordings to avoid duplicates
UNKNOWN_SAVE_COOLDOWN = 10
TOLERANCE = 0.5
CAMERA_INDEX = 0


def load_known_faces(dataset_dir):
    """Load labeled face encodings from the dataset directory."""
    encodings = []
    names = []
    for file_name in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = face_recognition.load_image_file(file_path)
        face_encs = face_recognition.face_encodings(image)
        if face_encs:
            encodings.append(face_encs[0])
            names.append(os.path.splitext(file_name)[0])
    return encodings, names


def alert_known(name):
    """Simple alert when a known person is recognized."""
    print(f"Recognized: {name}")
    print("\a", end="")  # attempt to beep


def save_unknown(frame, video_capture):
    """Save the current frame and a short video clip for an unknown face."""
    os.makedirs(UNKNOWN_IMAGE_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_VIDEO_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(UNKNOWN_IMAGE_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(image_path, frame)

    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(UNKNOWN_VIDEO_DIR, f"unknown_{ts}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for _ in range(int(fps * VIDEO_CLIP_SECONDS)):
        ret, clip_frame = video_capture.read()
        if not ret:
            break
        writer.write(clip_frame)
    writer.release()
    print(f"Unknown person recorded: {image_path}, {video_path}")


def recognize():
    known_encodings, known_names = load_known_faces(DATASET_DIR)
    if not known_encodings:
        print("No known faces loaded. Populate the dataset directory with images.")

    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        print("Cannot open camera.")
        return

    try:
        last_unknown_time = datetime.min
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = small[:, :, ::-1]

            locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locations)

            unknown_present = False
            for face_encoding in encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
                name = "Unknown"
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_index = np.argmin(distances)
                    if matches and matches[best_index]:
                        name = known_names[best_index]

                if name == "Unknown":
                    unknown_present = True
                else:
                    alert_known(name)

            if unknown_present:
                now = datetime.now()
                if (now - last_unknown_time).total_seconds() > UNKNOWN_SAVE_COOLDOWN:
                    save_unknown(frame, video_capture)
                    last_unknown_time = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
