import os
import cv2
import sys
import argparse
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detectors import ClockDetector
from src.core import settings

def test_video_detection(args):
    # Check if input is camera index (int) or video file
    video_input = args.video
    try:
        cam_index = int(video_input)
        is_camera = True
    except ValueError:
        is_camera = False

    if not is_camera and not os.path.exists(video_input):
        print(f"Error: Video not found at: {video_input}")
        return

    print(f"⏳ Loading Model from {args.model}...")
    detector = ClockDetector(model_path=args.model, conf_threshold=args.conf)

    # Open video or camera
    cap = cv2.VideoCapture(cam_index if is_camera else video_input)
    if not cap.isOpened():
        print("Error: Cannot open video/camera.")
        return

    print("🎬 Processing... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Detect keypoints
        keypoints = detector.detect(frame)

        # Draw keypoints if detected
        if keypoints is not None:
            for i, (x, y) in enumerate(keypoints.astype(int)):
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Video/Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Video/Camera Detection with ClockDetector")
    parser.add_argument('--model', type=str, default='best.pt', help='Path to the YOLO model file.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file or camera index (e.g., 0 for webcam).')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for detection.')

    args = parser.parse_args()
    test_video_detection(args)
