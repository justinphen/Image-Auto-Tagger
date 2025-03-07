import torch
import re

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class VisionLanguageModel:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                        device_map="auto"
                    )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    def identify_object(self, image, text_prompt="What is this object?"):
        # Ensure the image is a PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Calculate the maximum resolution based on the aspect ratio
        aspect_ratio = image.width / image.height
        max_pixels = 1_003_520  # Upper limit for the recommended token range

        # Calculate target width and height
        target_width = int((max_pixels * aspect_ratio) ** 0.5)
        target_height = int((max_pixels / aspect_ratio) ** 0.5)

        # Resize the image to the target resolution
        target_resolution = (target_width, target_height)
        image = image.resize(target_resolution, Image.Resampling.LANCZOS)  # Use high-quality resampling

        # Debug: Check the image size and type
        print(f"Image type: {type(image)}")
        print(f"Image size: {image.size}")

        # Prepare input for the VLM
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Debug: Check the inputs
        print(f"Inputs keys: {inputs.keys()}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        object_name = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return object_name