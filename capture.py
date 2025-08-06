import os
import cv2

DATASET_DIR = "dataset"
CAMERA_INDEX = 0

def capture_faces(num_photos: int = 3) -> None:
    """Capture face images for a given person.

    The user is prompted to enter a name. Images are stored under
    ``dataset/<name>`` with sequential numbering. For each image, the user
    confirms the capture by pressing the Enter key while a preview window is
    displayed. Press ``q`` to abort capturing for the current person.

    Parameters
    ----------
    num_photos: int, optional
        Number of photos to capture per person. Defaults to 3.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"Trying to open camera at index {CAMERA_INDEX}")
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    print("Camera opened successfully!")

    try:
        while True:
            name = input("请输入姓名（直接回车退出）：").strip()
            if not name:
                break
            person_dir = os.path.join(DATASET_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            existing = [
                f
                for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            count = 0
            while count < num_photos:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                cv2.imshow("Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
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
