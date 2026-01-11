import os
import cv2
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detectors import ClockDetector
from src.utils.geometry import four_point_transform
from src.core import settings

def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def crop_clock(args):
    if not os.path.exists(args.video):
        print(f"Error: Video not found at: {args.video}")
        return
    
    print(f"⏳ Loading Model from {args.model}...")
    detector = ClockDetector(model_path=args.model, conf_threshold=args.conf)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    
    print("🎬 Processing... Press 'q' to quit.")
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        frame_count += 1

        # Only process and save every 3rd frame
        if frame_count % 3 == 0:
            # Detect keypoints
            keypoints = detector.detect(frame)

            # If detected, warp and save the cropped clock image
            if keypoints is not None:
                try:
                    warped_img = four_point_transform(frame, keypoints, settings.warp_size)

                    blur_score = get_blur_score(warped_img)
                    
                    if blur_score > args.blur_threshold:
                        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                        output_path = os.path.join(args.output_dir, f"clock_{timestamp}ms.jpg")
                        saved_count += 1

                        cv2.imwrite(output_path, warped_img)
                        print(f"✅ Saved (Score: {blur_score:.1f}): {output_path}")

                    else:
                        print(f"❌ Skipped Blurry Image (Score: {blur_score:.1f} < {args.blur_threshold})")
                except Exception as e:
                    print(f"Error during warping: {e}")

        cv2.imshow("Video Cropping", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Clock from Video using ClockDetector")
    parser.add_argument('--model', type=str, default='best.pt', help='Path to the YOLO model file.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='cropped_clocks', help='Directory to save cropped clock images.')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for detection.')
    parser.add_argument('--blur_threshold', type=float, default=70.0, help='Blur score threshold to filter images.')

    args = parser.parse_args()
    crop_clock(args)
