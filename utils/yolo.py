import numpy as np

from PIL import Image
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self):
        # Load the YOLO model
        self.model = YOLO("models/yolov8m.pt")

    def detect_objects(self, image):
        # Ensure the image is in the correct format (PIL or numpy array)
        if isinstance(image, Image.Image):
            image = np.array(image)  # Convert PIL image to numpy array

        # Perform object detection
        results = self.model(image)

        # Extract bounding boxes in [x1, y1, x2, y2] format
        boxes = results[0].boxes.xyxy.cpu().numpy()
        return boxes