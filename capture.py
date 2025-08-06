import os
import subprocess

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
        # 使用rpicam-still捕获图像
        cmd = [
            "rpicam-still", 
            "-o", output_path,
            "--width", "640",
            "--height", "480",
            "--timeout", "1000"  # 1秒超时
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
    """
    print("使用 rpicam-still 进行图像捕获")
    
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
            while count < num_photos:
                input(f"准备拍摄第 {count + 1}/{num_photos} 张照片，按回车键拍摄...")
                
                count += 1
                idx = len(existing) + count
                file_path = os.path.join(person_dir, f"{idx}.jpg")
                
                print("正在拍摄...")
                if capture_with_rpicam(file_path):
                    print(f"✓ 已保存 {file_path}")
                else:
                    print("✗ 拍摄失败，请重试")
                    count -= 1  # 重试这张照片
                    
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