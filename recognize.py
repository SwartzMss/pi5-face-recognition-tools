import os
import cv2
import face_recognition
import numpy as np
from collections import deque
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


def save_unknown(frames, video_capture, fps, frame_size):
    """Save buffered frames and additional footage for an unknown face."""
    os.makedirs(UNKNOWN_IMAGE_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_VIDEO_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(UNKNOWN_IMAGE_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(image_path, frames[-1])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(UNKNOWN_VIDEO_DIR, f"unknown_{ts}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # Write pre-trigger frames from the buffer
    for buf_frame in frames:
        writer.write(buf_frame)

    # Record additional frames after trigger
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

    # Prepare video properties and buffering
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    frame_buffer = deque(maxlen=int(fps * VIDEO_CLIP_SECONDS))

    try:
        last_unknown_time = datetime.min
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Maintain a rolling buffer of recent frames
            frame_buffer.append(frame.copy())

            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = small[:, :, ::-1]

            locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locations)

            unknown_present = False
            if encodings:
                encodings_array = np.array(encodings)
                if known_encodings:
                    known_array = np.array(known_encodings)
                    distance_matrix = np.linalg.norm(
                        known_array[:, None, :] - encodings_array[None, :, :], axis=2
                    )
                    best_match_indices = np.argmin(distance_matrix, axis=0)
                    best_match_distances = distance_matrix[
                        best_match_indices, np.arange(len(encodings))
                    ]
                    for i, distance in enumerate(best_match_distances):
                        if distance <= TOLERANCE:
                            alert_known(known_names[best_match_indices[i]])
                        else:
                            unknown_present = True
                else:
                    unknown_present = True

            if unknown_present:
                now = datetime.now()
                if (now - last_unknown_time).total_seconds() > UNKNOWN_SAVE_COOLDOWN:
                    save_unknown(list(frame_buffer), video_capture, fps, frame_size)
                    last_unknown_time = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
