#!/usr/bin/env python3
"""
简单的GUI测试脚本，用于诊断OpenCV显示问题
"""
import cv2
import numpy as np
import subprocess
import os
import time

def test_opencv_gui():
    """测试OpenCV GUI功能"""
    print("=== OpenCV GUI 测试 ===")
    
    # 创建一个简单的测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (0, 100, 0)  # 绿色背景
    cv2.putText(test_image, "OpenCV GUI Test", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    try:
        print("尝试显示测试窗口...")
        cv2.imshow("Test Window", test_image)
        print("✓ 窗口创建成功")
        
        print("等待5秒或按任意键继续...")
        key = cv2.waitKey(5000)
        print(f"按键代码: {key}")
        
        cv2.destroyAllWindows()
        print("✓ GUI测试完成")
        return True
        
    except Exception as e:
        print(f"✗ GUI测试失败: {e}")
        return False

def test_rpicam_capture():
    """测试rpicam-still图像捕获"""
    print("\n=== rpicam-still 测试 ===")
    
    temp_image = "/tmp/test_capture.jpg"
    
    try:
        cmd = [
            "rpicam-still",
            "-o", temp_image,
            "--width", "640",
            "--height", "480",
            "--timeout", "1000"
        ]
        
        print("执行rpicam-still命令...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and os.path.exists(temp_image):
            print("✓ rpicam-still 捕获成功")
            
            # 尝试用OpenCV读取
            frame = cv2.imread(temp_image)
            if frame is not None:
                print(f"✓ 图像读取成功: {frame.shape}, dtype: {frame.dtype}")
                
                # 尝试显示
                try:
                    cv2.imshow("RPiCam Test", frame)
                    print("✓ rpicam图像显示成功")
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"✗ rpicam图像显示失败: {e}")
                
                # 清理
                os.remove(temp_image)
                return True
            else:
                print("✗ 无法读取捕获的图像")
                return False
        else:
            print(f"✗ rpicam-still 失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ rpicam测试异常: {e}")
        return False

def check_display_environment():
    """检查显示环境"""
    print("\n=== 显示环境检查 ===")
    
    # 检查DISPLAY变量
    display = os.environ.get('DISPLAY')
    print(f"DISPLAY: {display}")
    
    # 检查是否有X11
    try:
        result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ X11显示服务正常")
        else:
            print("✗ X11显示服务异常")
    except:
        print("✗ 没有X11显示服务")
    
    # 检查OpenCV构建信息
    print(f"OpenCV版本: {cv2.__version__}")
    
    # 检查GUI后端
    try:
        backends = cv2.getBuildInformation()
        if 'GTK' in backends:
            print("✓ 支持GTK后端")
        if 'QT' in backends:
            print("✓ 支持QT后端")
    except:
        print("无法获取OpenCV构建信息")

if __name__ == "__main__":
    print("树莓派5 GUI诊断工具")
    print("=" * 40)
    
    # 1. 检查显示环境
    check_display_environment()
    
    # 2. 测试OpenCV GUI
    gui_ok = test_opencv_gui()
    
    # 3. 测试rpicam捕获
    rpicam_ok = test_rpicam_capture()
    
    print("\n=== 诊断结果 ===")
    print(f"OpenCV GUI: {'✓ 正常' if gui_ok else '✗ 异常'}")
    print(f"RPiCam捕获: {'✓ 正常' if rpicam_ok else '✗ 异常'}")
    
    if not gui_ok:
        print("\n建议:")
        print("1. 确保在图形界面环境下运行")
        print("2. 尝试: export DISPLAY=:0")
        print("3. 检查X11权限: xhost +")
        print("4. 安装GUI支持: sudo apt install python3-opencv")