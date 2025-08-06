import os
import cv2
import time

# 数据集目录
DATASET_DIR = "dataset"
# CSI 摄像头在 V4L2 下的设备路径
CAMERA_DEVICE = "/dev/video0"


def capture_faces(num_photos: int = 3) -> None:
    """
    Capture face images for a given person.

    The user is prompted to enter a name. Images are stored under
    ``dataset/<name>`` with sequential numbering. For each image, the user
    confirms the capture by pressing the Enter key while a preview window is
    displayed. Press ``q`` to abort capturing for the current person.

    Parameters
    ----------
    num_photos: int, optional
        Number of photos to capture per person. Defaults to 3.
    """
    # 打开 V4L2 设备
    print(f"Opening camera device {CAMERA_DEVICE} with V4L2 backend…")
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: 无法打开摄像头设备 {CAMERA_DEVICE}")
        # 尝试不使用后端指定
        print("Trying without backend specification...")
        cap = cv2.VideoCapture(CAMERA_DEVICE)
        if not cap.isOpened():
            # 尝试其他常见设备路径
            for device_path in ["/dev/video1", "/dev/video2"]:
                print(f"Trying alternative device {device_path}...")
                cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                if cap.isOpened():
                    break
            else:
                print("ERROR: 无法打开任何摄像头设备")
                print("请检查:")
                print("1. 摄像头是否正确连接")
                print("2. 是否有足够的权限访问摄像头设备")
                print("3. 运行 'ls -la /dev/video*' 查看可用设备")
                return
    print("Camera opened successfully!")

    # 等待摄像头初始化
    time.sleep(2)
    print("Camera initialization completed.")

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 验证摄像头参数
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera parameters set: {width}x{height} @ {fps} FPS")
    
    # 摄像头预热 - 读取几帧让摄像头稳定
    print("Warming up camera...")
    for i in range(10):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)
    
    # 测试读取一帧
    print("Testing frame capture...")
    ret, test_frame = cap.read()
    if ret:
        print("✓ Frame capture test successful")
        print(f"Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
    else:
        print("✗ Frame capture test failed - camera may not be working properly")
        print("摄像头可能需要更多时间初始化，建议检查硬件连接")

    try:
        while True:
            name = input("请输入姓名（直接回车退出）：").strip()
            if not name:
                break
            person_dir = os.path.join(DATASET_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            existing = [
                f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            count = 0
            retry_count = 0
            max_retries = 5
            while count < num_photos:
                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    print(f"Failed to grab frame (attempt {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        print("摄像头读取失败次数过多，请检查摄像头连接")
                        break
                    time.sleep(0.5)  # 等待一下再重试
                    continue
                retry_count = 0  # 成功读取后重置重试计数
                cv2.imshow("Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                # Enter 键确认拍照
                if key in (13, ord('\r')):
                    count += 1
                    idx = len(existing) + count
                    file_path = os.path.join(person_dir, f"{idx}.jpg")
                    cv2.imwrite(file_path, frame)
                    print(f"已保存 {file_path}")
                    cv2.waitKey(300)  # small pause to avoid double captures
                elif key == ord('q'):
                    break
            cont = input("是否继续添加其他人? (y/n): ").strip().lower()
            if cont != 'y':
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    num_input = input("输入每人拍摄照片数量 [3]: ").strip()
    photos = int(num_input) if num_input.isdigit() else 3
    capture_faces(photos)
