import gradio as gr

from PIL import Image, ImageDraw

class GradioInterface:
    def __init__(self, annotation, object_detection, region_segmentation, vlm_model):
        self.annotation = annotation
        self.object_detection = object_detection
        self.region_segmentation = region_segmentation
        self.vlm_model = vlm_model

    def generate_response(self, annotation_format, image, prompts):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Object Detection
        bounding_boxes = self.object_detection.detect_objects(image)

        # Region Segmentation
        masks, segmented_images = self.region_segmentation.segment(image, bounding_boxes)

        # Object Identification
        object_names = []
        for segmented_image in segmented_images:
            object_name = self.vlm_model.identify_object(segmented_image)
            object_names.append(object_name)

        # Annotation Generation
        draw = ImageDraw.Draw(image)
        annotations = []
        image_width, image_height = image.size

        for i, (mask, object_name) in enumerate(zip(masks, object_names)):
            # Get bounding box from the mask
            box = self._get_bounding_box(mask)

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), object_name, fill="red")

            # Generate annotations
            if annotation_format == "YOLO":
                annotations.append(self.annotation_utils.to_yolo_format(box, image_width, image_height))
            elif annotation_format == "COCO":
                annotations.append(self.annotation_utils.to_coco_format(box, image_id=1, annotation_id=i + 1))
            elif annotation_format == "CVAT":
                annotations.append(self.annotation_utils.to_cvat_format(box, object_name))

        # Format annotations based on user selection
        if annotation_format == "YOLO":
            annotations_output = "\n".join(annotations)
        elif annotation_format == "COCO":
            annotations_output = annotations
        elif annotation_format == "CVAT":
            annotations_output = "\n".join(annotations)

        return image, annotations_output

    def _get_bounding_box(self, mask):
        mask_array = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        coords = np.where(mask_array > 0)
        x1, y1 = coords[1].min(), coords[0].min()
        x2, y2 = coords[1].max(), coords[0].max()
        return (x1, y1, x2, y2)

    def create_interface(self):
        demo = gr.Interface(
            fn=self.generate_response,
            inputs=[
                gr.Radio(["YOLO", "COCO", "CVAT"], label="Annotation Format"),
                gr.Image(type="pil"),
                gr.Textbox(label="Prompts")
            ],
            outputs=[
                gr.Image(label="Annotated Image"),
                gr.Textbox(label="Annotations"),
            ],
            title="Image Autotagger",
            description="Upload an image, enter prompts, and select an annotation format to get an annotated image and annotations.",
        )
        return demo