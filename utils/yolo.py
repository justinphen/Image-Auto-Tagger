from ultralytics import YOLO

class ObjectDetection:
    def __init__(self):
        self.model = self._load_yolo_model()

    def _load_yolo_model(self):
        return YOLO("models/yolov8m.pt")

    def detect_objects(self, image):
        results = self.model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        class_names = self.model.names
        return boxes, labels, class_names