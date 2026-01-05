from ultralytics import YOLO

class DigitDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        try:
            self.model = YOLO(model_path)
            self.conf =  conf_threshold
        except Exception as e:
            print(f"Error: {e}")
            self.model = None

    def detect(self, image):
        if self.model  is None or image is None:
            return []
        
        results = self.model.predict(image, conf=self.conf, verbose=False)

        if len(results) == 0:
            return []
        
        # Format: [x_min, y_min, x_max, y_max, confidence, class_id]
        boxes = results[0].boxes.data.cpu().numpy()

        detected_items = []

        for box in boxes:
            x_min = box[0]
            cls_id = int(box[5])

            class_name = self.model.names[cls_id]

            detected_items.append((x_min, class_name))

        detected_items.sort(key=lambda x: x[0])

        final_digits = [item[1] for item in detected_items]

        return final_digits