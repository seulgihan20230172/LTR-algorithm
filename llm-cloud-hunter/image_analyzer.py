'''
from image_classifier import ImageClassifier
from image_transcriber import ImageTranscriber


class ImageAnalyzer:
    def __init__(self, image_classifier: ImageClassifier = None, image_transcriber: ImageTranscriber = None):
        self.image_classifier = image_classifier if image_classifier else ImageClassifier()
        self.image_transcriber = image_transcriber if image_transcriber else ImageTranscriber()

    def analyze_image(self, paragraph: str, number_of_images: int, image_index: int, image_url: str) -> tuple[str, str] | None:
        image_classification = self.image_classifier.classify_image(paragraph, number_of_images, image_index, image_url)
        if image_classification.informative:
            image_transcription = self.image_transcriber.transcribe_image(paragraph, number_of_images, image_index, image_url)
            return image_classification.description.replace('**', '*'), image_transcription.replace('**', '*')
        return None
'''
from image_classifier import ImageClassifier
from image_transcriber import ImageTranscriber
import logging


class ImageAnalyzer:
    def __init__(self, image_classifier: ImageClassifier = None, image_transcriber: ImageTranscriber = None):
        self.image_classifier = image_classifier if image_classifier else ImageClassifier()
        self.image_transcriber = image_transcriber if image_transcriber else ImageTranscriber()

    def analyze_image(self, paragraph: str, number_of_images: int, image_index: int, image_url: str) -> tuple[str, str] | None:
        image_classification = self.image_classifier.classify_image(
            paragraph, number_of_images, image_index, image_url
        )

        # ✅ None 방어 (분류 실패/타임아웃/파싱 실패 등)
        if image_classification is None:
            logging.warning(f"\t\tImage classification returned None. Skipping. url={image_url}")
            return None

        # ✅ attribute 방어 (혹시 객체 구조가 예상과 다를 때)
        if not getattr(image_classification, "informative", False):
            return None

        image_transcription = self.image_transcriber.transcribe_image(
            paragraph, number_of_images, image_index, image_url
        )

        # transcribe_image도 실패할 수 있으니 방어
        if image_transcription is None:
            logging.warning(f"\t\tImage transcription returned None. url={image_url}")
            image_transcription = ""

        desc = getattr(image_classification, "description", "")
        return desc.replace('**', '*'), str(image_transcription).replace('**', '*')
