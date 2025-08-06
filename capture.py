import os
import subprocess
import time

# 数据集目录
DATASET_DIR = "dataset"


def capture_with_rpicam(output_path: str) -> bool:
    """
    使用 rpicam-still 命令捕获图像 (树莓派5)
    
    Parameters
    ----------
    output_path: str
        输出图像文件路径
        
    Returns
    -------
    bool
        是否成功捕获图像
    """
    try:
        # 使用rpicam-still捕获图像，优化参数
        cmd = [
            "rpicam-still", 
            "-o", output_path,
            "--width", "640",
            "--height", "480",
            "--timeout", "2000",     # 2秒超时
            "--brightness", "0.3",   # 增加亮度
            "--contrast", "1.2",     # 增强对比度
            "--saturation", "1.0",   # 正常饱和度
            "--awb", "tungsten",     # 钨丝灯白平衡，减少绿色调
            "--ev", "0.5",           # 增加曝光补偿
            "--quality", "95",       # 高质量JPEG
            "--encoding", "jpg"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        else:
            print(f"rpicam-still failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("rpicam-still timeout")
        return False
    except FileNotFoundError:
        print("rpicam-still command not found")
        return False
    except Exception as e:
        print(f"rpicam-still error: {e}")
        return False



def capture_faces(num_photos: int = 3) -> None:
    """
    使用 rpicam-still 命令进行人脸图像捕获 (树莓派5)
    带实时预览，按回车拍照
    """
    print("使用 rpicam-still 进行图像捕获")
    print("预览窗口将持续显示，在终端按回车键拍照")
    
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
            
            # 启动持续预览
            print(f"正在为 {name} 启动摄像头预览...")
            print("在终端按回车键拍照，按 Ctrl+C 结束当前人员拍摄")
            
            preview_process = None
            try:
                # 启动预览进程（后台运行）
                preview_cmd = [
                    "rpicam-hello",
                    "--timeout", "0",        # 持续预览
                    "--width", "640",
                    "--height", "480", 
                    "--brightness", "0.3",
                    "--contrast", "1.2",
                    "--saturation", "1.0",
                    "--awb", "tungsten",
                    "--ev", "0.5"
                ]
                preview_process = subprocess.Popen(preview_cmd, 
                                                 stdout=subprocess.DEVNULL, 
                                                 stderr=subprocess.DEVNULL)
                
                # 等待预览启动
                time.sleep(2)
                print("预览已启动，现在可以开始拍照了！")
                
                count = 0
                while count < num_photos:
                    input(f"拍摄第 {count + 1}/{num_photos} 张照片 - 按回车键拍摄...")
                    
                    # 拍照前停止预览
                    print("正在拍摄...")
                    if preview_process:
                        preview_process.terminate()
                        preview_process.wait()
                    
                    count += 1
                    idx = len(existing) + count
                    file_path = os.path.join(person_dir, f"{idx}.jpg")
                    
                    # 拍照
                    if capture_with_rpicam(file_path):
                        print(f"✓ 已保存 {file_path}")
                    else:
                        print("✗ 拍摄失败，请重试")
                        count -= 1  # 重试这张照片
                    
                    # 如果还有照片要拍，重新启动预览
                    if count < num_photos:
                        print("重新启动预览...")
                        preview_process = subprocess.Popen(preview_cmd, 
                                                         stdout=subprocess.DEVNULL, 
                                                         stderr=subprocess.DEVNULL)
                        time.sleep(1)  # 等待预览启动
                        
            except KeyboardInterrupt:
                print(f"\n{name} 拍摄结束")
            finally:
                # 结束预览进程
                if preview_process:
                    preview_process.terminate()
                    preview_process.wait()
                    print("预览已关闭")
                    
            cont = input("是否继续添加其他人? (y/n): ").strip().lower()
            if cont != 'y':
                break
                
    except KeyboardInterrupt:
        print("\n拍摄已取消")
        return


if __name__ == "__main__":
    num_input = input("输入每人拍摄照片数量 [3]: ").strip()
    photos = int(num_input) if num_input.isdigit() else 3
    capture_faces(photos)