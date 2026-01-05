from ultralytics import YOLO
import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core import logger

def test_model(args, mode='detection'):
    # 1. Validate file paths
    if not os.path.exists(args.weights):
        logger.error(f"Critical Error: Weights file not found at {args.weights}")
        return
    if not os.path.exists(args.data):
        logger.error(f"Critical Error: Data file not found at {args.data}")
        return

    try:
        # 2. Load the trained model
        model = YOLO(args.weights)

        # 3. Run Testing
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,       
            iou=args.iou,         
            split=args.split,    
            project=f'runs/{mode}_val', 
            name=args.name,
            exist_ok=True,
            plots=True            
        )

        # 4. Log key metrics
        logger.info(f"Testing {mode} completed!")
        
        if mode == 'detection':
            # Log Pose Metrics (mAP50-95 for keypoints)
            logger.info(f"mAP50-95 (Pose): {metrics.box.map:.4f}") 
            logger.info(f"mAP50 (Pose):    {metrics.box.map50:.4f}")
        else:
            # Log Detection Metrics (Digits)
            logger.info(f"mAP50-95 (Box):  {metrics.box.map:.4f}")
            logger.info(f"mAP50 (Box):     {metrics.box.map50:.4f}")
            
        logger.info(f"Results saved to: runs/{mode}_val/{args.name}")

    except Exception as e:
        logger.error(f"Testing Failed: {e}")
        raise e

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Evaluate/Test Digital Clock Models")
    subparsers = parser.add_subparsers(dest='selection', required=True, help='Choose testing mode')

    def add_common_test_args(sub_p):
        sub_p.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
        sub_p.add_argument('--weights', type=str, required=True, help='Path to trained .pt file')
        sub_p.add_argument('--batch', type=int, default=16, help="Batch size")
        sub_p.add_argument('--device', default='0', help='Device to run on (0, 1, 2 or cpu)')
        sub_p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
        sub_p.add_argument('--iou', type=float, default=0.6, help='NMS IoU threshold')
        sub_p.add_argument('--split', type=str, default='val', choices=['val', 'test'], help="Dataset split to use ('val' or 'test')")

    # Detection
    parser_det = subparsers.add_parser('detection', help='Test Pose Model (Find Clock)')
    parser_det.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser_det.add_argument('--name', default='eval_clock_pose', help='Name of the results folder')
    add_common_test_args(parser_det)

    # Recognition
    parser_rec = subparsers.add_parser('recognition', help='Test Recognition Model (Read Digits)')
    parser_rec.add_argument('--imgsz', type=int, default=320, help='Input image size')
    parser_rec.add_argument('--name', default='eval_digit_rec', help='Name of the results folder')
    add_common_test_args(parser_rec)

    args = parser.parse_args()

    test_model(args, mode=args.selection)