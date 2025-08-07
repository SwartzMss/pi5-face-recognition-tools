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
    直接拍照，无预览窗口
    """
    # 1. 配置并启动摄像头
    picam2 = Picamera2()
    
    # 简化的摄像头配置
    config = picam2.create_preview_configuration(
        main={"size": (640, 480)},    # 主流，用于高质量拍摄
        lores={"size": (320, 240)}    # 预览流
    )
    picam2.configure(config)
    
    # 设置基本的摄像头参数
    picam2.set_controls({
        "AeEnable": True,           # 自动曝光
        "AwbEnable": True,          # 自动白平衡
        "Brightness": 0.0,          # 亮度
        "Contrast": 1.0,            # 对比度
        "Saturation": 1.0,          # 饱和度
    })
    
    picam2.start()

    try:
        print("使用 Picamera2 进行图像捕获")
        print("提示：使用 'rpicam-hello' 命令可以查看摄像头预览")
        print("在终端按回车键拍照，按 Ctrl+C 退出")

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

                try:
                    # 捕获 main 流并保存
                    rgb = picam2.capture_array("main")
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    idx = len(existing_files) + count + 1
                    file_path = os.path.join(person_dir, f"{idx}.jpg")
                    cv2.imwrite(file_path, frame)
                    print(f"✓ 已保存: {file_path}")

                    count += 1
                    time.sleep(0.5)  # 给缓冲一点时间
                except Exception as e:
                    print(f"拍照错误: {e}")
                    continue

            cont = input("是否继续添加其他人? (y/n): ").strip().lower()
            if cont != 'y':
                break

    except KeyboardInterrupt:
        print("\n捕获已取消，正在退出...")

    finally:
        # 停止摄像头
        picam2.close()
        print("摄像头已关闭")


if __name__ == "__main__":
    num_input = input("输入每人拍摄照片数量 [3]: ").strip()
    photos = int(num_input) if num_input.isdigit() else 3
    capture_faces(photos)