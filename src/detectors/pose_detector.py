import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path, conf_threshold=0.4):
        print(f"Loading model pose from: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.conf  = conf_threshold
        except Exception as e:
            print(f"Error: {e}")
            raise e
        
    def detect(self, image):
        results = self.model.predict(image, conf=self.conf, verbose=False)

        if len(results) == 0:
            return None
        
        r = results[0]

        if r.keypoints is None or r.keypoints.data.shape[1] == 0:
            return None
        
        kpts = r.keypoints.xy[0].cpu().numpy()

        if np.all(kpts ==  0):
            return None
        
        return kpts