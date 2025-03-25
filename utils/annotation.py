import numpy as np

class AnnotationFormats:
    @staticmethod
    def to_yolo_format(box, image_width, image_height):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        return f"0 {x_center} {y_center} {width} {height}"

    @staticmethod
    def to_coco_format(mask, image_id, annotation_id):
        # Ensure mask is binary (0 = background, 1 = object)
        mask = mask.astype(np.uint8)

        # Get nonzero pixel coordinates
        y_indices, x_indices = np.where(mask > 0)  

        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Compute bounding box coordinates
        x1, y1, x2, y2 = np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)
        width, height = x2 - x1, y2 - y1

        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "segmentation": mask.tolist(),
            "iscrowd": 0,
        }

    @staticmethod
    def to_cvat_format(box, label):
        x1, y1, x2, y2 = box
        return f"{label} {x1} {y1} {x2} {y2}"