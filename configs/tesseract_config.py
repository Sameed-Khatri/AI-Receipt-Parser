import os
from dotenv import load_dotenv
import pytesseract
from PIL import Image


load_dotenv()

class OCR:
    def __init__(self):
        self.__tesseract_path = os.getenv("TESSERACT_PATH")
        pytesseract.pytesseract.tesseract_cmd = self.__tesseract_path


    def _normalize_boxes(self, ocr_data, image):
        boxes = []

        width, height = image.size
        for i in range(len(ocr_data["text"])):
            if ocr_data["text"][i].strip() == "":
                continue
            (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])

            boxes.append([
                int(1000 * x / width),
                int(1000 * y / height),
                int(1000 * (x + w) / width),
                int(1000 * (y + h) / height)
            ])

        return boxes
    

    def _extract_text(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        data = pytesseract.image_to_data(image=image, output_type=pytesseract.Output.DICT)
        
        return data, image
    

    async def run_ocr(self, image_path: str):
        data, image = self._extract_text(image_path=image_path)
        boxes = self._normalize_boxes(ocr_data=data, image=image)
        words = [w for w in data["text"] if w.strip() != ""]

        return {
            "words": words,
            "boxes": boxes,
            "image": image
        }