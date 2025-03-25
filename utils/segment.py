import cv2
import os
import torch
import numpy as np

from PIL import Image
from models.sam2.sam2.build_sam import build_sam2
from models.sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class RegionSegmentation:
    def __init__(self):
        """
        Initializes the RegionSegmentation class by loading Meta's SAM2 model.
        """
        sam2_checkpoint = "models/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.device = "cuda"

        self.sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     model=self.sam2,
        #     points_per_side=16,
        #     points_per_batch=16,
        #     pred_iou_thresh=0.7,
        #     stability_score_thresh=0.92,
        #     stability_score_offset=0.7,
        #     crop_n_layers=1,
        #     box_nms_thresh=0.7,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=25.0,
        #     use_m2m=True,
        # )

        # Create output directory if it doesn't exist
        self.output_dir = "segmented_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_segmented_images(self, image, masks):
        """
        Extracts segmented objects from the given masks.

        Args:
            image (PIL.Image): Original image.
            masks (list of dict): List of masks from SAM2.

        Returns:
            list: A list of cropped segmented images.
        """
        segmented_images = []
        image_np = np.array(image)

        for idx, ann in enumerate(masks):
        # for ann in masks:
            mask = ann["segmentation"].astype(np.uint8)

            # Find contours of the segmented regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Get bounding box around the object
                x, y, w, h = cv2.boundingRect(contours[0])

                # Extract and store the segmented object
                segmented_object = image_np[y:y+h, x:x+w]
                segmented_images.append(segmented_object)

                # Convert to PIL format and save
                save_path = os.path.join(self.output_dir, f"segment_{idx}.png")
                Image.fromarray(segmented_object).save(save_path)

        return segmented_images

    def run_sam2(self, image):
        """
        Runs SAM2 automatic segmentation on an image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            list: List of segmented images.
        """
        image_np = np.array(image)

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks = self.mask_generator.generate(image_np)

        # Get segmented images
        segmented_images = self.get_segmented_images(image_np, masks)

        return segmented_images
