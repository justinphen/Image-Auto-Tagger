import torch
import numpy as np

from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

class RegionSegmentation:
    def __init__(self):
        self.model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    def segment(self, image, boxes):
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Get image dimensions
        height, width = image.shape[:2]

        # Set the image for the predictor
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.set_image(image)
            
            # SAM2 expects boxes in the format [x1, y1, x2, y2] with normalized coordinates
            normalized_boxes = boxes / np.array([width, height, width, height])

            # Generate masks using the bounding box prompts
            masks, _, _ = self.model.predict(box=normalized_boxes)

        # Create segmented images from the masks
        segmented_images = []
        for mask in masks:
            # Ensure mask is a numpy array
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            # Handle multi-channel masks
            if mask.ndim == 3:  # If the mask has multiple channels (e.g., RGB)
                mask = mask[0]  # Use the first channel (or another channel if needed)

            # Ensure mask is 2D
            if mask.ndim != 2:
                raise ValueError(f"Mask has invalid shape: {mask.shape}. Expected 2D array.")

            # Convert to 0-255 range
            mask = (mask * 255).astype(np.uint8)

            # Convert the mask to a PIL.Image object
            mask_image = Image.fromarray(mask, mode="L")  # 'L' mode for grayscale
            segmented_images.append(mask_image)

        return masks, segmented_images