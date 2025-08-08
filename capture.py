import os
import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2 import Preview

# 数据集目录
DATASET_DIR = "dataset"


def enhance_image(frame):
    """增强图像质量
    
    应用图像增强技术来改善图像质量
    """
    # 1. 颜色校正 - 解决绿蓝色偏问题
    # 转换为LAB颜色空间
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对a和b通道进行白平衡校正
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a = cv2.add(a, 128 - a_mean)
    b = cv2.add(b, 128 - b_mean)
    
    # 对L通道进行CLAHE（对比度限制自适应直方图均衡）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 合并通道并转换回BGR
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. 颜色饱和度增强
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # 增加饱和度
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 3. 降噪处理
    # 使用非局部均值去噪
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # 4. 锐化处理 - 增强边缘细节
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 5. 对比度增强
    # 转换为YUV颜色空间进行亮度调整
    yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # 对Y通道进行直方图均衡
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 6. 最终的双边滤波 - 保持边缘的同时平滑
    enhanced = cv2.bilateralFilter(enhanced, 15, 80, 80)
    
    return enhanced


def capture_faces(num_photos: int = 3) -> None:
    """
    使用 Picamera2 API 进行人脸图像捕获 (树莓派5)
    直接拍照，无预览窗口
    """
    # 1. 配置并启动摄像头
    picam2 = Picamera2()
    
    # 创建更高质量的配置
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},    # 主流，用于高质量拍摄
        lores={"size": (640, 480)}     # 预览流
    )
    picam2.configure(config)
    
    # 设置优化的摄像头参数
    picam2.set_controls({
        "AeEnable": True,           # 自动曝光
        "AwbEnable": True,          # 自动白平衡
        "AeMeteringMode": 0,        # 平均测光
        "AeExposureMode": 0,        # 自动曝光模式
        "AwbMode": 0,               # 自动白平衡模式
        "Brightness": 0.0,          # 亮度
        "Contrast": 1.2,            # 对比度增强
        "Saturation": 1.3,          # 饱和度增强
        "Sharpness": 1.5,           # 锐度增强
        "NoiseReductionMode": 1,    # 降噪模式
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

                time.sleep(2)
                try:
                    # 捕获 main 流并保存
                    rgb = picam2.capture_array("main")
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    
                    # 暂时关闭图像增强，直接保存原始图像
                    # enhanced_frame = enhance_image(frame)

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