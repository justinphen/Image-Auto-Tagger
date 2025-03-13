import torch
import numpy as np

from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

class RegionSegmentation:
    def __init__(self):
        """
        Initializes the RegionSegmentation class by loading Meta's SAM2 model.
        """
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    def get_segmented_images(self, image, masks):
        """
        Extracts and saves segmented objects from the given masks.

        Args:
            image (PIL.Image): Original image.
            masks (list of np.ndarray): List of binary masks.

        Returns:
            list: A list of cropped segmented images.
        """
        segmented_images = []
        image_np = np.array(image)

        for i, mask in enumerate(masks):
            # Ensure mask is in 2D
            mask = mask.squeeze()

            # Convert mask to binary
            mask = (mask > 0.5).astype(np.uint8) * 255

            # Find bounding box of the mask
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                # Skip empty masks
                continue

            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # Extract the masked object
            masked_object = image_np * (mask[..., None] // 255)

            # Crop the masked object
            cropped_object = masked_object[y_min:y_max, x_min:x_max]

            # Append the segmented image
            segmented_images.append(cropped_object)

        return segmented_images
    
    def run_sam2(self, bounding_boxes, image):
        """
        Runs SAM2 segmentation on an image using bounding boxes from YOLO.

        Args:
            image (PIL.Image): Input image.

        Returns:
            list: List of segmented images.
        """
        # Run SAM2 segmentation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(np.array(image))
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                # Convert list to array
                box=np.array(bounding_boxes)[None, :],
                multimask_output=False
            )

        # Get segmented images
        segmented_images = self.get_segmented_images(image, masks)
        
        return segmented_images