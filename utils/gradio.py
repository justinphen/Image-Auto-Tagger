import torch
import gradio as gr
import numpy as np

from PIL import Image, ImageDraw

class GradioInterface:
    def __init__(self, annotation, region_segmentation, vlm_model):
        """
        Initializes the GradioInterface class.

        Args:
        - annotation: An instance of the Annotation class for converting bounding boxes to various formats.
        - region_segmentation: An instance of the RegionSegmentation class for segmenting objects in images.
        - vlm_model: An instance of the VisionLanguageModel class for identifying objects in segmented images.
        """
        self.annotation = annotation
        self.region_segmentation = region_segmentation
        self.vlm_model = vlm_model

    def generate_response(self, annotation_format, image):
        """
        Processes an image to segment regions, identify objects, and generate annotations.

        Args:
            annotation_format (str): The format for annotations (e.g., "YOLO", "COCO", "CVAT").
            image (PIL.Image or np.ndarray): Input image in PIL or NumPy array format.

        Returns:
            str or list: Annotations in the specified format.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Region Segmentation
        segmented_images = self.region_segmentation.run_sam2(image)

        # Object Identification
        object_names = []
        for segmented_image in segmented_images:
            object_name = self.vlm_model.identify_object(segmented_image)
            if isinstance(object_name, bytes):
                object_name = object_name.decode("utf-8")
            elif isinstance(object_name, torch.Tensor):
                object_name = object_name.item() if object_name.numel() == 1 else str(object_name)
            object_names.append(object_name)

        # Annotation Generation
        annotations = []
        image_width, image_height = image.size

        for i, (segmented_image, object_name) in enumerate(zip(segmented_images, object_names)):
            x1, y1, x2, y2 = 0, 0, segmented_image.shape[1], segmented_image.shape[0]

            if annotation_format == "YOLO":
                annotations.append(self.annotation.to_yolo_format([x1, y1, x2, y2], image_width, image_height))
            elif annotation_format == "COCO":
                annotations.append(self.annotation.to_coco_format([x1, y1, x2, y2], image_id=1, annotation_id=i + 1))
            elif annotation_format == "CVAT":
                annotations.append(self.annotation.to_cvat_format([x1, y1, x2, y2], object_name))

        # Format annotations based on user selection
        if annotation_format in ["YOLO", "CVAT"]:
            annotations_output = "\n".join(annotations)
        elif annotation_format == "COCO":
            annotations_output = annotations

        return annotations_output

    def create_interface(self):
        """
        Creates a Gradio interface for the image autotagger.

        Returns:
            gr.Interface: A Gradio interface object.
        """
        demo = gr.Interface(
            fn=self.generate_response,
            inputs=[
                gr.Radio(["YOLO", "COCO", "CVAT"], label="Annotation Format"),
                gr.Image(type="pil")
            ],
            outputs=[
                gr.Textbox(label="Annotations"),
            ],
            title="Image Autotagger",
            description="Upload an image, enter prompts, and select an annotation format to get annotations.",
        )
        return demo