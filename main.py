import cv2
import os
import sys

sys.path.append(os.getcwd())

from src.core import settings
from src.services.clock_reader import ClockReader

def main():
    if settings is None: return
    
    if not os.path.exists(settings.pose_model_path):
        print(f"Error: {settings.pose_model_path}")
        return
    
    print(f"{settings.pose_model_path}...")
    try: 
        clock_service = ClockReader(
        pose_model_path=settings.pose_model_path,
        digit_model_path=settings.digit_model_path, 
        pose_conf=settings.pose_conf,
        digit_conf=settings.digit_conf
    )
    except Exception as e:
        print(f"Error: {e}")
        return
    
    cap = cv2.VideoCapture(settings.camera_id)
    if  not cap.isOpened():
        print(f"Error: {settings.camera_id}")
        return
    
    while True:
        ret, frame = cap.read()
        if  not ret:
            break

        debug_frame, warped_img, time_text = clock_service.process_frame(frame)

        cv2.imshow("Main View", debug_frame)

        if  warped_img is not None:
            cv2.imshow("Warped  Output", warped_img)

            if time_text !=  "...":
                print(f"Detected Time: {time_text}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    main()

