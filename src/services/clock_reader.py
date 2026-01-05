import cv2
import numpy as np
from src.detectors import PoseDetector, DigitDetector
from src.utils.geometry import four_point_transform

class ClockReader:
    def __init__(self, pose_model_path, digit_model_path, pose_conf=0.6, digit_conf=0.5):
        self.pose_detector = PoseDetector(pose_model_path, pose_conf)
        self.digit_detector = DigitDetector(digit_model_path, digit_conf)

    def process_frame(self, frame):
        keypoints = self.pose_detector.detect(frame)

        warped_img = None
        time_text = "Checking..."
        debug_frame = frame.copy()

        if keypoints is not None:
            try: 
                warped_img  = four_point_transform(frame, keypoints)

                digits = self.digit_detector.detect(warped_img)

                time_text = "".join(digits) if digits else "..."

                pts_int = keypoints.astype(int)
                
                for i, (x, y) in enumerate(pts_int):
                    cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)
                
                pts_poly = pts_int.reshape((-1, 1, 2))
                cv2.polylines(debug_frame, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=2)

                text_pos = (pts_int[0][0], pts_int[0][1] - 10)
                cv2.putText(debug_frame, f"Result: {time_text}", text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            except Exception as e:
                print(f"Error; {e}")

        return debug_frame, warped_img