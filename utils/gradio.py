import gradio as gr

from PIL import ImageDraw

class GradioInterface:
    def __init__(self, vlm_model, yolo_model, annotation_utils):
        self.vlm_model = vlm_model
        self.yolo_model = yolo_model
        self.annotation_utils = annotation_utils

    def generate_response(self, image, text_prompt, annotation_format):
        if image is None or text_prompt.strip() == "":
            return "Please provide both an image and a text prompt."

        # Use VLM to identify relevant objects
        output_text = self.vlm_model.generate_response(image, text_prompt)
        relevant_objects = self.vlm_model.extract_relevant_objects(output_text)

        # Run YOLO detection and filter for relevant objects
        boxes, labels, class_names = self.yolo_model.detect_objects(image)

        # Draw bounding boxes and generate annotations
        draw = ImageDraw.Draw(image)
        annotations = []
        image_width, image_height = image.size

        for box, label in zip(boxes, labels):
            label_text = class_names[int(label)]
            if label_text.lower() in relevant_objects:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 10), label_text, fill="red")

                # Generate annotations
                if annotation_format == "YOLO":
                    annotations.append(self.annotation_utils.to_yolo_format(box, image_width, image_height))
                elif annotation_format == "COCO":
                    annotations.append(self.annotation_utils.to_coco_format(box, image_id=1, annotation_id=len(annotations) + 1))
                elif annotation_format == "CVAT":
                    annotations.append(self.annotation_utils.to_cvat_format(box, label_text))

        # Format annotations based on user selection
        if annotation_format == "YOLO":
            annotations_output = "\n".join(annotations)
        elif annotation_format == "COCO":
            annotations_output = annotations
        elif annotation_format == "CVAT":
            annotations_output = "\n".join(annotations)

        return image, annotations_output

    def create_interface(self):
        return gr.Interface(
            fn=self.generate_response,
            inputs=[
                gr.Image(type="pil"),
                gr.Textbox(label="Text Prompt"),
                gr.Radio(["YOLO", "COCO", "CVAT"], label="Annotation Format"),
            ],
            outputs=[
                gr.Image(label="Annotated Image"),
                gr.Textbox(label="Annotations"),
            ],
            title="Image Autotagger",
            description="Upload an image, enter a text prompt, and select an annotation format to get an annotated image and annotations.",
        )