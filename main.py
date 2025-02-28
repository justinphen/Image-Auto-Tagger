from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO
import gradio as gr
import numpy as np
import torch

def autotag_image(image, tags, threshold=0.2):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Process the image and text prompts
    inputs = processor(text=tags, images=image, return_tensors="pt", padding=True)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Filter tags based on the threshold
    predicted_tags = [tags[i] for i, prob in enumerate(probs[0]) if prob > threshold]
    return predicted_tags

def process_image(image, text_prompt, annotation_format):
    # Load YOLO model
    yolo_model = YOLO("yolov8m.pt")

    # Detect objects
    yolo_results = yolo_model(image)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    labels = yolo_results[0].boxes.cls.cpu().numpy()

    # Draw bounding boxes on the image
    image_np = np.array(image)
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"Class {int(label)}", fill="red")

    # Generate annotations in the selected format
    annotations = []
    image_width, image_height = image.size
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if annotation_format == "YOLO":
            annotations.append(to_yolo_format(box, image_width, image_height))
        elif annotation_format == "COCO":
            annotations.append(to_coco_format(box, image_id=1, annotation_id=i + 1))
        elif annotation_format == "CVAT":
            annotations.append(to_cvat_format(box, f"Class {int(label)}"))

    return image, "\n".join(annotations) if annotation_format != "COCO" else annotations

def to_yolo_format(box, image_width, image_height):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"0 {x_center} {y_center} {width} {height}"

def to_coco_format(box, image_id, annotation_id):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x1, y1, width, height],
        "area": width * height,
        "iscrowd": 0,
    }

def to_cvat_format(box, label):
    x1, y1, x2, y2 = box
    return f"{label} {x1} {y1} {x2} {y2}"

def run_app(image, text_prompt, annotation_format):
    annotated_image, annotations = process_image(image, text_prompt, annotation_format)
    return annotated_image, annotations


interface = gr.Interface(
    fn=run_app,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Text Prompt"),
        gr.Radio(["YOLO", "COCO", "CVAT"], label="Annotation Format"),
    ],
    outputs=[
        gr.Image(label="Annotated Image"),
        gr.Textbox(label="Annotations"),
    ],
    title="Image Autotagger"
)

interface.launch(share=True)