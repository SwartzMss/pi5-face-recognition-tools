import os
import cv2
import face_recognition
import numpy as np

from datetime import datetime
import threading
import time
from picamera2 import Picamera2

DATASET_DIR = "dataset"  # 数据集目录
UNKNOWN_IMAGE_DIR = "unknown_images"  # 未知人脸图像保存目录
# Cooldown between unknown recordings to avoid duplicates
UNKNOWN_SAVE_COOLDOWN = 10  # 保存未知人脸的冷却时间，避免重复记录
TOLERANCE = 0.5  # 人脸识别的距离阈值


class PiCamera2Stream:
    """使用 Picamera2 API 实现视频流"""
    
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.latest_frame = None
        self.picam2 = None
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        
    def capture_frames(self):
        """持续从Picamera2捕获帧"""
        print("启动Picamera2视频流...")
        
        try:
            while self.running:
                try:
                    # 捕获RGB数组并转换为BGR（OpenCV格式）
                    rgb_frame = self.picam2.capture_array()
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    
                    with self.frame_lock:
                        self.latest_frame = bgr_frame.copy()
                        
                    # 控制帧率
                    time.sleep(1.0 / self.fps)
                    
                except Exception as e:
                    print(f"捕获帧错误: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"视频流捕获线程错误: {e}")
                
    def start_stream(self):
        """启动视频流"""
        print("启动Picamera2视频流...")
        try:
            self.picam2 = Picamera2()
            
            # 创建更高质量的配置
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={
                    "FrameDurationLimits": (33333, 33333),  # 30fps
                    "AeEnable": True,  # 自动曝光
                    "AwbEnable": True,  # 自动白平衡
                    "AeMeteringMode": 0,  # 平均测光
                    "AeExposureMode": 0,  # 自动曝光模式
                    "AwbMode": 0,  # 自动白平衡模式
                    "Brightness": 0.0,  # 亮度
                    "Contrast": 1.0,  # 对比度
                    "Saturation": 1.0,  # 饱和度
                    "Sharpness": 1.0,  # 锐度
                    "NoiseReductionMode": 1,  # 降噪模式
                }
            )
            self.picam2.configure(config)
            
            # 设置相机参数
            self.picam2.set_controls({
                "AeEnable": True,
                "AwbEnable": True,
                "AeMeteringMode": 0,
                "AeExposureMode": 0,
                "AwbMode": 0,
                "Brightness": 0.0,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 1.0,
                "NoiseReductionMode": 1,
            })
            
            self.picam2.start()
            
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.capture_thread.start()
            
            # 等待第一帧
            print("等待视频流初始化...")
            for i in range(50):  # 最多等待5秒
                with self.frame_lock:
                    if self.latest_frame is not None:
                        print("✓ Picamera2视频流启动成功")
                        return self
                time.sleep(0.1)
                
            print("✗ 视频流启动失败")
            self.stop_stream()
            return None
            
        except Exception as e:
            print(f"启动Picamera2失败: {e}")
            return None
    
    def read(self):
        """读取最新帧，模拟OpenCV VideoCapture.read()"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return False, None
    
    def isOpened(self):
        """检查流是否打开"""
        return self.running and (self.picam2 is not None)
    
    def release(self):
        """停止并清理"""
        self.stop_stream()
    
    def stop_stream(self):
        """停止视频流"""
        print("停止Picamera2视频流...")
        self.running = False
        
        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        # 停止摄像头
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None
    
    def get(self, prop):
        """模拟OpenCV的get方法"""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0





def load_known_faces(dataset_dir):
    """Load labeled face encodings from the dataset directory.

    从数据集目录中加载带标签的人脸编码。`dataset_dir` 下应该为每个人
    创建一个子目录，目录名作为标签，目录中可以包含多张该人的照片。
    如果某张图片中检测到多张人脸，将会跳过并给出警告，以避免错误
    的标签影响识别准确率。
    """
    encodings = []
    names = []
    print(f"开始加载训练数据，目录: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        return encodings, names
    
    person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"找到 {len(person_dirs)} 个人员目录: {person_dirs}")
    
    for person_name in person_dirs:
        person_dir = os.path.join(dataset_dir, person_name)
        print(f"处理人员: {person_name}, 目录: {person_dir}")
        
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  找到 {len(image_files)} 张图片: {image_files}")
        
        for file_name in image_files:
            file_path = os.path.join(person_dir, file_name)
            print(f"  处理图片: {file_path}")
            
            try:
                image = face_recognition.load_image_file(file_path)
                print(f"    图片加载成功，形状: {image.shape}")

                # 确保图像为连续内存，避免 dlib 接口报错
                image = np.ascontiguousarray(image)

                face_encs = face_recognition.face_encodings(image)
                print(f"    检测到 {len(face_encs)} 个人脸特征")
                
                if len(face_encs) == 1:
                    encodings.append(face_encs[0])  # 只使用检测到的第一张人脸
                    names.append(person_name)  # 目录名作为标签
                    print(f"    ✓ 成功添加 {person_name} 的人脸特征")
                elif len(face_encs) > 1:
                    print(f"    ⚠️  警告: 在 {file_path} 中检测到多个人脸，跳过此图片")
                else:
                    print(f"    ❌ 错误: 在 {file_path} 中未检测到人脸")
                    
            except Exception as e:
                print(f"    ❌ 处理图片 {file_path} 时出错: {e}")
    
    print(f"总共加载了 {len(encodings)} 个人脸编码")
    return encodings, names


def alert_known(name):
    """Simple alert when a known person is recognized.

    当识别到已知人员时给出提示
    """
    print(f"Recognized: {name}")
    print("\a", end="")  # 尝试发出蜂鸣声


def enhance_image(frame):
    """增强图像质量
    
    应用图像增强技术来改善图像质量
    """
    # 转换为LAB颜色空间进行白平衡调整
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道进行CLAHE（对比度限制自适应直方图均衡）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 合并通道
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 应用双边滤波来减少噪声同时保持边缘
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 轻微锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced


def save_unknown(frame):
    """Save the current frame for an unknown face.

    保存当前帧作为未知人脸的记录
    """
    # 创建按日期分类的目录结构
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    
    # 创建日期目录
    date_dir = os.path.join(UNKNOWN_IMAGE_DIR, date_str)
    os.makedirs(date_dir, exist_ok=True)
    
    # 增强图像质量
    enhanced_frame = enhance_image(frame)
    
    # 使用时分秒作为文件名
    image_path = os.path.join(date_dir, f"{time_str}.jpg")
    cv2.imwrite(image_path, enhanced_frame)  # 保存增强后的帧
    print(f"Unknown person recorded: {image_path}")


def save_known(frame, name):
    """Save the current frame for a known person.

    保存当前帧作为已知人脸的记录
    """
    # 增强图像质量
    enhanced_frame = enhance_image(frame)
    
    # 使用年月日时分秒作为文件名
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(UNKNOWN_IMAGE_DIR, f"{name}_{ts}.jpg")
    cv2.imwrite(image_path, enhanced_frame)  # 保存增强后的帧
    print(f"Known person recorded: {image_path}")


def recognize():
    """Main loop for real-time face recognition.

    实时进行人脸识别的主循环
    """
    known_encodings, known_names = load_known_faces(DATASET_DIR)
    print(f"加载了 {len(known_encodings)} 个已知人脸编码")
    if not known_encodings:
        print("No known faces loaded. Populate the dataset directory with images.")

    # 使用PiCamera2Stream代替传统VideoCapture
    video_capture = PiCamera2Stream(1280, 720, 30)  # 提高分辨率和帧率
    cap = video_capture.start_stream()
    if cap is None:
        print("无法启动Picamera2视频流")
        return

    # 摄像头已在启动时预热，无需额外预热
    print("Camera ready for recognition...")
    
    # Prepare video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:  # 添加FPS上限检查
        fps = 15  # 使用PiCamera2Stream的默认FPS
        print(f"Using default FPS: {fps}")
    else:
        print(f"Camera FPS: {fps}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    last_unknown_time = datetime.min
    last_known_time = datetime.min
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
            ret, frame = cap.read()
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



            # 增强图像质量
            enhanced_frame = enhance_image(frame)
            
            # 使用增强后的图像进行人脸检测
            # 将 BGR 转为 RGB，并确保连续内存
            rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            # 调试：检查图像数据
            if frame_count % 30 == 0:  # 每30帧打印一次调试信息
                print(f"Frame shape: {frame.shape}, rgb shape: {rgb.shape}")
                print(f"Frame dtype: {frame.dtype}, rgb dtype: {rgb.dtype}")
                print(f"Frame range: {frame.min()}-{frame.max()}, rgb range: {rgb.min()}-{rgb.max()}")

            locations = face_recognition.face_locations(rgb)  # 找到人脸位置
            encodings = face_recognition.face_encodings(rgb)  # 计算人脸特征

            # 调试信息
            if frame_count % 10 == 0:  # 每10帧打印一次检测结果
                print(f"检测到 {len(locations)} 个人脸位置，{len(encodings)} 个人脸特征")
                if locations:
                    print(f"人脸位置: {locations}")

            unknown_present = False
            known_present = False
            face_names = []
            known_names_detected = []
            
            # 确保 locations 和 encodings 匹配
            if locations and encodings:
                # 只处理有特征的人脸
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
                            known_present = True
                            known_names_detected.append(name)
                        else:
                            name = "Unknown"
                            unknown_present = True
                        face_names.append(name)
                else:
                    face_names = ["Unknown"] * len(encodings)
                    unknown_present = True

            # Draw results on the frame
            for i, (top, right, bottom, left) in enumerate(locations):
                # 获取对应的名字（如果有的话）
                name = face_names[i] if i < len(face_names) else "Unknown"
                
                # 设置颜色：已知人员为绿色，未知人员为红色
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(enhanced_frame, (left, top), (right, bottom), color, 2)
                
                # 添加标签背景
                label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(enhanced_frame, (left, top - 35), (left + label_size[0], top), color, -1)
                cv2.putText(
                    enhanced_frame,
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
                    save_unknown(frame)
                    last_unknown_time = now  # 更新上次记录时间
            
            # 保存已知人脸的记录
            if known_present:
                now = datetime.now()
                if (now - last_known_time).total_seconds() > UNKNOWN_SAVE_COOLDOWN:
                    # 保存第一个检测到的已知人脸
                    save_known(frame, known_names_detected[0])
                    last_known_time = now  # 更新上次记录时间

            # 调试信息
            if frame is not None:

                # 在帧上添加状态信息
                status_text = f"FPS: {frame_count / (current_time - start_time):.1f} | Faces: {len(face_names)}"
                cv2.putText(enhanced_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(enhanced_frame, "Press 'q' to quit", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 尝试显示窗口
                try:
                    cv2.imshow("Face Recognition - Pi5", enhanced_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break  # 按 q 键退出
                except Exception as e:
                    print(f"GUI display error: {e}")
                    print("继续无GUI模式运行...")
                    # 无GUI模式，只在终端显示识别结果
                    if face_names:
                        print(f"检测到人脸: {', '.join(face_names)}")
            else:
                print("警告: 获取到空帧")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
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
