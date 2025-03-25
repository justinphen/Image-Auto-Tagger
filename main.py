from utils.annotation import AnnotationFormats
from utils.gradio import GradioInterface
from utils.segment import RegionSegmentation
from utils.vlm import VisionLanguageModel

class ImageAutotagger:
    def __init__(self):
        """
        Initializes the ImageAutotagger class.

        This class integrates the following components:
        - AnnotationFormats: For converting bounding boxes to various formats (YOLO, COCO, CVAT).
        - RegionSegmentation: For segmenting objects in images using SAM2.
        - VisionLanguageModel: For identifying objects in segmented images using Qwen2.5-VL.
        """
        self.annotation = AnnotationFormats()
        self.region_segmentation = RegionSegmentation()
        self.vlm = VisionLanguageModel()

        # Create Gradio interface
        self.gradio_interface = GradioInterface(self.annotation, self.region_segmentation, self.vlm)
        self.demo = self.gradio_interface.create_interface()

    def __call__(self):
        """
        Launches the Gradio interface for the Image Autotagger.
        """
        self.demo.launch(share=True, debug=True)

if __name__ == "__main__":
    agent = ImageAutotagger()
    agent()