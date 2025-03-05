import torch
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import re

# Load Qwen2.5-VL model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load YOLO model
yolo_model = YOLO("yolov8m.pt")

# Define the function for Gradio
def generate_response(image, text_prompt, annotation_format):
    if image is None or text_prompt.strip() == "":
        return "Please provide both an image and a text prompt."

    # Use VLM to identify relevant objects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Prepare input for Qwen2.5-VL
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response VLM
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1280)

    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Extract relevant objects from VLM response
    relevant_objects = extract_relevant_objects(output_text)

    # Run YOLO detection and filter for relevant objects
    yolo_results = yolo_model(image)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    labels = yolo_results[0].boxes.cls.cpu().numpy()
    class_names = yolo_model.names

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
                annotations.append(to_yolo_format(box, image_width, image_height))
            elif annotation_format == "COCO":
                annotations.append(to_coco_format(box, image_id=1, annotation_id=len(annotations) + 1))
            elif annotation_format == "CVAT":
                annotations.append(to_cvat_format(box, label_text))

    # Format annotations based on user selection
    if annotation_format == "YOLO":
        annotations_output = "\n".join(annotations)
    elif annotation_format == "COCO":
        annotations_output = annotations
    elif annotation_format == "CVAT":
        annotations_output = "\n".join(annotations)

    return image, annotations_output

def extract_relevant_objects(response_text):
    # Extract relevant objects from Qwen2.5-VL's response
    print(response_text)
    objects = re.findall(r'\b[a-zA-Z]+\b', response_text.lower())
    print(objects)
    return set(objects)

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

# Gradio Interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Text Prompt"),
        gr.Radio(["YOLO", "COCO", "CVAT"], label="Annotation Format"),
    ],
    outputs=[
        gr.Image(label="Annotated Image"),
        gr.Textbox(label="Annotations"),
    ],
    title="Qwen2.5-VL + YOLO Image Annotation",
    description="Upload an image, enter a text prompt, and select an annotation format to get an annotated image and annotations.",
)

# Run the Gradio app
demo.launch(share=True)