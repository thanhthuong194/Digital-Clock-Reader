import cv2
import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detectors import PoseDetector  
from src.utils.geometry import four_point_transform
from src.core import settings

def test_warp_logic(args):
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    # 3. Khởi tạo Detector
    print(f"⏳ Loading Model form {args.model}...")
    detector = PoseDetector(model_path=args.model, conf_threshold=0.4)

    # 4. Đọc ảnh
    original_img = cv2.imread(args.image)
    if original_img is None:
        print("Error: Cannot read image.")
        return

    # 5. Chạy Detect
    keypoints = detector.detect(original_img)

    if keypoints is not None:
        print("Phone Detected!")
        print(f"Keypoints coordinates:\n{keypoints}")

        # 6. Chạy Warp
        try:
            warped_img = four_point_transform(original_img, keypoints, settings.warp_size)
            
            # A. Vẽ 4 điểm lên ảnh gốc để đối chiếu
            debug_img = original_img.copy()

            for i, (x, y) in enumerate(keypoints.astype(int)):
                cv2.circle(debug_img, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(debug_img, str(i), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Vẽ khung nối các điểm
            pts_poly = keypoints.astype(int).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts_poly], isClosed=True, color=(0, 255, 255), thickness=2)

            # B. Hiển thị kết quả
            h, w = debug_img.shape[:2]
            if w > 800:
                scale = 800 / w
                debug_img = cv2.resize(debug_img, (0,0), fx=scale, fy=scale)

            cv2.imshow("1. Detected Keypoints (Original)", debug_img)
            cv2.imshow("2. Warped Result (Cropped)", warped_img)
            
            print("👀 Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error during warping: {e}")
    else:
        print("⚠️ No phone detected in this image. Try another one.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pose Detection + Perspective Transform")
    
    parser.add_argument('--image', type=str, default='74cebec6-ac87-477a-a425-20fe3e472b49.jpg', help='Path to test image')
    parser.add_argument('--model', type=str, default='runs/detection/clock_pose/weights/best.pt', help='Path to .pt model')
    
    args = parser.parse_args()
    test_warp_logic(args)