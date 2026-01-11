import numpy as np
from ultralytics import YOLO

class ClockDetector:
    def __init__(self, model_path, conf_threshold):
        print(f"Loading model pose from: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.conf  = conf_threshold
        except Exception as e:
            print(f"Error: {e}")
            raise e
        
    def detect(self, image):
        try:
            results = self.model.predict(image, conf=self.conf, verbose=False)

            if len(results) == 0:
                return None
            
            result = results[0]
        

            if result.keypoints is None or result.keypoints.data.shape[1] == 0:
                return None
            
            kpts = result.keypoints.xy[0].cpu().numpy()

            if np.all(kpts == 0):
                return None
            return kpts
        except Exception as e:
            print(f"Detect error: {e}")
            return None