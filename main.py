from utils.annotation import AnnotationFormats
from utils.gradio import GradioInterface
from utils.yolo import ObjectDetection
from utils.segment import RegionSegmentation
from utils.vlm import VisionLanguageModel

class ImageAutotagger:
    def __init__(self):
        # Initialize models and utilities
        self.annotation = AnnotationFormats()
        self.object_detection = ObjectDetection()
        self.region_segmentation = RegionSegmentation()
        self.vlm = VisionLanguageModel()

        # Create Gradio interface
        self.gradio_interface = GradioInterface(self.annotation, self.object_detection, self.region_segmentation, self.vlm)
        self.demo = self.gradio_interface.create_interface()

    def __call__(self):
        self.demo.launch(share=True)

if __name__ == "__main__":
    agent = ImageAutotagger()
    agent()