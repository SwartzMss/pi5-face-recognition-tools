import os
import cv2
import face_recognition
import numpy as np
from collections import deque
from datetime import datetime
import subprocess
import threading
import time

DATASET_DIR = "dataset"  # 数据集目录
UNKNOWN_IMAGE_DIR = "unknown_images"  # 未知人脸图像保存目录
UNKNOWN_VIDEO_DIR = "unknown_videos"  # 未知人脸视频保存目录
VIDEO_CLIP_SECONDS = 3  # 每次录制视频的秒数
# Cooldown between unknown recordings to avoid duplicates
UNKNOWN_SAVE_COOLDOWN = 10  # 保存未知人脸的冷却时间，避免重复记录
TOLERANCE = 0.5  # 人脸识别的距离阈值
CAMERA_INDEX = 0  # 摄像头索引
MAX_CAMERA_INDEX = 10  # 尝试搜索摄像头的最大数量


def open_camera_pi5():
    """Open camera using libcamera for Pi5 compatibility.
    
    为树莓派5打开摄像头，优先使用libcamera，回退到V4L2
    """
    # 先尝试 libcamera (Pi5 推荐)
    print("Trying libcamera backend...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        if cap.isOpened():
            # 测试读取帧
            ret, _ = cap.read()
            if ret:
                print("✓ Libcamera backend working")
                # 设置参数
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            else:
                print("✗ Libcamera backend failed to read frames")
                cap.release()
    except Exception as e:
        print(f"Libcamera attempt failed: {e}")
    
    # 回退到 V4L2
    print("Trying V4L2 backend...")
    for device in ["/dev/video0", "/dev/video1", "/dev/video2"]:
        try:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"✓ V4L2 backend working with {device}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
                cap.release()
        except Exception as e:
            print(f"V4L2 attempt with {device} failed: {e}")
    
    print("ERROR: 无法打开任何摄像头设备")
    print("请检查:")
    print("1. 摄像头是否正确连接")
    print("2. 运行 'rpicam-hello' 测试摄像头是否工作")
    return None


def open_available_camera(preferred_index=CAMERA_INDEX, max_index=MAX_CAMERA_INDEX):
    """Wrapper function for backward compatibility."""
    return open_camera_pi5()


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

    # 摄像头预热
    print("Warming up camera...")
    for i in range(10):
        ret, _ = video_capture.read()
        if ret:
            break
        time.sleep(0.1)
    
    # Prepare video properties and buffering
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:  # 添加FPS上限检查
        fps = 30  # 如果无法获取FPS或FPS异常，使用默认值30
        print(f"Using default FPS: {fps}")
    else:
        print(f"Camera FPS: {fps}")
        
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    frame_size = (width, height)
    frame_buffer = deque(maxlen=int(fps * VIDEO_CLIP_SECONDS))

    last_unknown_time = datetime.min
    frame_read_failures = 0
    max_failures = 5
    
    # 性能监控
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    
    try:
        print("Starting face recognition loop...")
        print("Press 'q' to quit")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                frame_read_failures += 1
                print(f"Failed to read frame from camera (attempt {frame_read_failures}/{max_failures})")
                if frame_read_failures >= max_failures:
                    print("Too many consecutive frame read failures. Exiting.")
                    break
                time.sleep(0.1)  # 短暂等待后重试
                continue
            
            frame_read_failures = 0  # 成功读取后重置失败计数
            frame_count += 1

            # 每5秒显示一次FPS
            current_time = time.time()
            if current_time - last_fps_time >= 5.0:
                actual_fps = frame_count / (current_time - start_time)
                print(f"Processing FPS: {actual_fps:.1f}")
                last_fps_time = current_time

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
                # 缩放回原始尺寸
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # 设置颜色：已知人员为绿色，未知人员为红色
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # 添加标签背景
                label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (left, top - 35), (left + label_size[0], top), color, -1)
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            if unknown_present:
                now = datetime.now()
                if (now - last_unknown_time).total_seconds() > UNKNOWN_SAVE_COOLDOWN:
                    save_unknown(list(frame_buffer), video_capture, fps, frame_size)
                    last_unknown_time = now  # 更新上次记录时间

            # 在帧上添加状态信息
            status_text = f"FPS: {frame_count / (current_time - start_time):.1f} | Faces: {len(face_names)}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Face Recognition - Pi5", frame)  # 显示识别结果
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # 按 q 键退出
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    print("=== 树莓派5 人脸识别系统 ===")
    print("确保已使用 capture.py 收集训练数据")
    
    # 检查是否有训练数据
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"警告: 未找到训练数据目录 '{DATASET_DIR}' 或目录为空")
        print("请先运行 capture.py 收集人脸数据")
        choice = input("是否继续运行? (y/n): ").strip().lower()
        if choice != 'y':
            sys.exit(0)
    
    recognize()
