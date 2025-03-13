import numpy as np

from PIL import Image
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self):
        """
        Initializes the ObjectDetection class by loading the YOLOv8 model.
        """
        self.model = YOLO("models/yolov8m.pt")

    def detect_objects(self, image):
        """
        Detects objects in the given image using the YOLOv8 model.

        Args:
            image (PIL.Image or np.ndarray): Input image in PIL or NumPy array format.

        Returns:
            np.ndarray: A NumPy array of bounding boxes in [x1, y1, x2, y2] format.
        """
        # Ensure the image is in the correct format (PIL or numpy array)
        if isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            image = np.array(image)

        # Perform object detection
        results = self.model(image)
        # Extract bounding boxes in [x1, y1, x2, y2] format
        boxes = results[0].boxes.xyxy.cpu().numpy()
        return boxes