import os
import threading
import time
import cv2
from picamera2 import Picamera2
from picamera2 import Preview

# 数据集目录
DATASET_DIR = "dataset"


def capture_faces(num_photos: int = 3) -> None:
    """
    使用 Picamera2 API 进行人脸图像捕获 (树莓派5)
    带实时预览，按回车拍照
    """
    # 1. 配置并启动摄像头
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480)},    # 主流，用于高质量拍摄
        lores={"size": (320, 240)}    # 预览流
    )
    picam2.configure(config)
    picam2.start()

    # 2. 启动预览窗口线程
    running = True

    def preview_loop():
        window_name = "Camera Preview"
        cv2.namedWindow(window_name)
        while running:
            # 获取 lores 流 RGB 数组
            rgb = picam2.capture_array("lores")
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(window_name)

    preview_thread = threading.Thread(target=preview_loop, daemon=True)
    preview_thread.start()

    try:
        print("使用 Picamera2 进行图像捕获")
        print("预览窗口已打开，在终端按回车键拍照，按 Ctrl+C 退出")

        while True:
            name = input("请输入姓名（直接回车退出）：").strip()
            if not name:
                break

            person_dir = os.path.join(DATASET_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            existing_files = [
                f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            count = 0
            while count < num_photos:
                input(f"[{name}] 拍摄第 {count + 1}/{num_photos} 张 - 按回车键拍照...")

                # 捕获 main 流并保存
                rgb = picam2.capture_array("main")
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                idx = len(existing_files) + count + 1
                file_path = os.path.join(person_dir, f"{idx}.jpg")
                cv2.imwrite(file_path, frame)
                print(f"✓ 已保存: {file_path}")

                count += 1
                time.sleep(0.5)  # 给缓冲一点时间

            cont = input("是否继续添加其他人? (y/n): ").strip().lower()
            if cont != 'y':
                break

    except KeyboardInterrupt:
        print("\n捕获已取消，正在退出...")

    finally:
        # 停止预览与摄像头
        running = False
        preview_thread.join(timeout=2)
        picam2.close()
        print("摄像头已关闭")


if __name__ == "__main__":
    num_input = input("输入每人拍摄照片数量 [3]: ").strip()
    photos = int(num_input) if num_input.isdigit() else 3
    capture_faces(photos)
