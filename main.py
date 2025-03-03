from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ultralytics import YOLO
import gradio as gr
import numpy as np
import torch
import re

# Load Qwen2.5VL
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Load YOLO model
yolo_model = YOLO("yolov8m.pt")

def process_image(image, text_prompt, annotation_format):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract object names from the response
    relevant_objects = extract_relevant_objects(response_text)

    # Run YOLO detection
    yolo_results = yolo_model(image)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    labels = yolo_results[0].boxes.cls.cpu().numpy()

    # Draw bounding boxes for relevant objects only
    image_np = np.array(image)
    draw = ImageDraw.Draw(image)
    annotations = []
    image_width, image_height = image.size

    for i, (box, label) in enumerate(zip(boxes, labels)):
        label_text = f"Class {int(label)}"
        if any(obj.lower() in label_text.lower() for obj in relevant_objects):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), label_text, fill="red")

            # Generate annotations
            if annotation_format == "YOLO":
                annotations.append(to_yolo_format(box, image_width, image_height))
            elif annotation_format == "COCO":
                annotations.append(to_coco_format(box, image_id=1, annotation_id=i + 1))
            elif annotation_format == "CVAT":
                annotations.append(to_cvat_format(box, label_text))

    return image, "\n".join(annotations) if annotation_format != "COCO" else annotations

def extract_relevant_objects(response_text):
    # Extract possible object names from Qwen2.5VL's response
    object_list = re.findall(r'\b[a-zA-Z]+(?:\s[a-zA-Z]+)?\b', response_text)
    return [obj.lower() for obj in object_list if len(obj) > 2]

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
    title="Qwen2.5VL Image Annotation"
)

interface.launch(share=True)