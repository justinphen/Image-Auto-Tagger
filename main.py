from utils.annotation import AnnotationFormats
from utils.gradio import GradioInterface
from utils.vlm import VisionLanguageModel
from utils.yolo import ObjectDetection

class ImageAutotagger:
    def __init__(self):
        """
        Initialize the main application with VLM, YOLO, and Gradio interface.
        """
        # Initialize models and utilities
        self.vlm = VisionLanguageModel()
        self.object_detection = ObjectDetection()
        self.annotation = AnnotationFormats()

        # Create Gradio interface
        self.gradio_interface = GradioInterface(self.vlm, self.object_detection, self.annotation)
        self.demo = self.gradio_interface.create_interface()

    def __call__(self):
        """
        Run the Gradio interface.
        """
        self.demo.launch(share=True)

if __name__ == "__main__":
    agent = ImageAutotagger()
    agent()