import torch
import re
import numpy as np

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VisionLanguageModel:
    def __init__(self):
        """
        Initializes the VisionLanguageModel class by loading the Qwen2.5-VL model and processor.
        
        - Model: "Qwen/Qwen2.5-VL-7B-Instruct"
        - Data type: torch.float16 (for efficiency)
        - Attention implementation: flash_attention_2 (for faster inference)
        - Device: Automatically mapped to available hardware (e.g., GPU or CPU).
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                        device_map="auto"
                    )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            min_pixels=256*28*28,
            max_pixels=1280*28*28
            )

    def identify_object(self, image):
        """
        Identifies the primary object in the given image using the Qwen2.5-VL model.

        Args:
            image (PIL.Image or np.ndarray): Input image in PIL or NumPy array format.

        Returns:
            str: The name of the identified object.

        Raises:
            ValueError: If the input image is None or not in a valid format.
        """
        # Ensure image is valid
        if image is None:
            raise ValueError("Received None as input image.")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL image or NumPy array.")

        # Define the expected input message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", 
                     "text": "Identify the object in this image and only return the object name."
                    },
                ],
            }
        ]

        # Apply chat template and process vision input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs for the model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate output from the model
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_length=4097)
            output_tokens_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_tokens)
            ]
            output_text = self.processor.batch_decode(
                output_tokens_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return output_text[0]