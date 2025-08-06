import os
import cv2
import face_recognition
import numpy as np
from collections import deque
from datetime import datetime

DATASET_DIR = "dataset"  # 数据集目录
UNKNOWN_IMAGE_DIR = "unknown_images"  # 未知人脸图像保存目录
UNKNOWN_VIDEO_DIR = "unknown_videos"  # 未知人脸视频保存目录
VIDEO_CLIP_SECONDS = 3  # 每次录制视频的秒数
# Cooldown between unknown recordings to avoid duplicates
UNKNOWN_SAVE_COOLDOWN = 10  # 保存未知人脸的冷却时间，避免重复记录
TOLERANCE = 0.5  # 人脸识别的距离阈值
CAMERA_INDEX = 0  # 摄像头索引
MAX_CAMERA_INDEX = 10  # 尝试搜索摄像头的最大数量


def open_available_camera(preferred_index=CAMERA_INDEX, max_index=MAX_CAMERA_INDEX):
    """Try to open a camera, scanning indices if needed.

    优先使用 `preferred_index`，若打开失败则从 0 开始依次尝试其它摄像头索引。
    返回成功打开的 `cv2.VideoCapture` 对象，如未找到可用摄像头则返回 None。
    """
    indices = []
    if preferred_index is not None:
        indices.append(preferred_index)
    indices.extend(i for i in range(max_index) if i != preferred_index)

    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            if idx != preferred_index:
                print(f"Using camera index {idx}")
            return cap
        cap.release()
    return None


def load_known_faces(dataset_dir):
    """Load labeled face encodings from the dataset directory.

    从数据集目录中加载带标签的人脸编码。`dataset_dir` 下应该为每个人
    创建一个子目录，目录名作为标签，目录中可以包含多张该人的照片。
    如果某张图片中检测到多张人脸，将会跳过并给出警告，以避免错误
    的标签影响识别准确率。
    """
    encodings = []
    names = []
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue  # 只处理子目录
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  # 只处理常见的图像文件
            file_path = os.path.join(person_dir, file_name)
            image = face_recognition.load_image_file(file_path)
            face_encs = face_recognition.face_encodings(image)
            if len(face_encs) == 1:
                encodings.append(face_encs[0])  # 只使用检测到的第一张人脸
                names.append(person_name)  # 目录名作为标签
            elif len(face_encs) > 1:
                print(
                    f"Warning: Multiple faces detected in {file_path}. Skipping this image."
                )
    return encodings, names


def alert_known(name):
    """Simple alert when a known person is recognized.

    当识别到已知人员时给出提示
    """
    print(f"Recognized: {name}")
    print("\a", end="")  # 尝试发出蜂鸣声


def save_unknown(frames, video_capture, fps, frame_size):
    """Save buffered frames and additional footage for an unknown face.

    将缓冲中的帧以及额外录制的帧保存下来，用于记录未知人脸
    """
    os.makedirs(UNKNOWN_IMAGE_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_VIDEO_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(UNKNOWN_IMAGE_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(image_path, frames[-1])  # 保存触发时刻的最后一帧

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(UNKNOWN_VIDEO_DIR, f"unknown_{ts}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # Write pre-trigger frames from the buffer
    for buf_frame in frames:
        writer.write(buf_frame)  # 写入触发前的缓存帧

    # Record additional frames after trigger
    for _ in range(int(fps * VIDEO_CLIP_SECONDS)):
        ret, clip_frame = video_capture.read()
        if not ret:
            break
        writer.write(clip_frame)  # 继续写入触发后的画面

    writer.release()
    print(f"Unknown person recorded: {image_path}, {video_path}")


def recognize():
    """Main loop for real-time face recognition.

    实时进行人脸识别的主循环
    """
    known_encodings, known_names = load_known_faces(DATASET_DIR)
    if not known_encodings:
        print("No known faces loaded. Populate the dataset directory with images.")

    video_capture = open_available_camera(CAMERA_INDEX)
    if video_capture is None:
        print("Cannot open camera.")
        return

    # Prepare video properties and buffering
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # 如果无法获取FPS，使用默认值30
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    frame_buffer = deque(maxlen=int(fps * VIDEO_CLIP_SECONDS))

    last_unknown_time = datetime.min
    try:
        print("Starting face recognition loop...")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break

            # Maintain a rolling buffer of recent frames
            frame_buffer.append(frame.copy())  # 保存最近的帧到缓冲区

            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 缩小图像以加快处理速度
            rgb_small = small[:, :, ::-1]  # 将 BGR 转为 RGB

            locations = face_recognition.face_locations(rgb_small)  # 找到人脸位置
            encodings = face_recognition.face_encodings(rgb_small, locations)  # 计算人脸特征

            unknown_present = False
            face_names = []
            if encodings:
                encodings_array = np.array(encodings)
                if known_encodings:
                    known_array = np.array(known_encodings)
                    distance_matrix = np.linalg.norm(
                        known_array[:, None, :] - encodings_array[None, :, :], axis=2
                    )  # 计算已知与未知人脸的距离矩阵
                    best_match_indices = np.argmin(distance_matrix, axis=0)
                    best_match_distances = distance_matrix[
                        best_match_indices, np.arange(len(encodings))
                    ]
                    for i, distance in enumerate(best_match_distances):
                        if distance <= TOLERANCE:
                            name = known_names[best_match_indices[i]]
                            alert_known(name)
                        else:
                            name = "Unknown"
                            unknown_present = True
                        face_names.append(name)
                else:
                    face_names = ["Unknown"] * len(encodings)
                    unknown_present = True

            # Draw results on the frame
            for (top, right, bottom, left), name in zip(locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # 绘制人脸矩形框
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )  # 在图像上写入名称

            if unknown_present:
                now = datetime.now()
                if (now - last_unknown_time).total_seconds() > UNKNOWN_SAVE_COOLDOWN:
                    save_unknown(list(frame_buffer), video_capture, fps, frame_size)
                    last_unknown_time = now  # 更新上次记录时间

            cv2.imshow("Video", frame)  # 显示识别结果
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # 按 q 键退出
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
