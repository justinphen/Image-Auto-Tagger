import torch
import gradio as gr
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# Load the model with flash attention for better performance
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Define the function for Gradio
def generate_response(image, text_prompt):
    if image is None or text_prompt.strip() == "":
        return "Please provide both an image and a text prompt."
    
    # Convert PIL image to the required format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Prepare input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1280)

    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

# Gradio Interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Image(type="pil"),  # Image input
        gr.Textbox(label="Text Prompt"),  # Text input
    ],
    outputs=gr.Textbox(label="VLM Response"),  # Text output
    title="Qwen2.5-VL Image Annotation",
    description="Upload an image and enter a text prompt to generate a response using Qwen2.5-VL.",
)

# Run the Gradio app
demo.launch(share=True)