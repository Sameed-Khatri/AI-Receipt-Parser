from typing import Dict
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


class Hunggingface:
    """
    wrapper class for hugging face fine tuned layoutlmv3 model and processor for token classification on receipts.

    attributes:
        __model_id (str): model identifier for the finet uned layoutlmv3.
        __processor (LayoutLMv3Processor): processor for image, text, and box encoding.
        __model (LayoutLMv3ForTokenClassification): token classification model instance.
    """

    def __init__(self):
        """
        initializes the hugging face model and processor for inference.
        """
        self.__model_id = "Sameed1/smdk-layoutlmv3-receipts"
        self.__processor = LayoutLMv3Processor.from_pretrained(self.__model_id)
        self.__model = LayoutLMv3ForTokenClassification.from_pretrained(self.__model_id)
        self.__model.to("cpu")
        self.__model.eval()
    

    def _generate_output(self, words, labels) -> Dict:
        """
        generates structured entity output from model predictions.

        args:
            words (List[str]): list of words from the receipt extracted using tesseract.
            labels (List[str]): corresponding predicted labels for each word.

        returns:
            dict: dictionary with extracted entities (company, date, address, total).
        """
        entities = {}
        current_label, current_tokens = None, []

        for word, label in zip(words, labels):
            if label == "O":
                if current_label:
                    entities.setdefault(current_label, []).append(" ".join(current_tokens))
                    current_label, current_tokens = None, []
                continue

            label_type = label.split("-")[-1]
            if label.startswith("B-"):
                if current_label:
                    entities.setdefault(current_label, []).append(" ".join(current_tokens))
                current_label, current_tokens = label_type, [word]
            elif label.startswith("I-") and current_label == label_type:
                current_tokens.append(word)

        if current_label:
            entities.setdefault(current_label, []).append(" ".join(current_tokens))

        result = {
            "company": entities.get("COMPANY", [""])[0],
            "date": entities.get("DATE", [""])[0],
            "address": entities.get("ADDRESS", [""])[0],
            "total": entities.get("TOTAL", [""])[0],
        }

        return result


    async def run_inference(self, image, words, boxes) -> Dict:
        """
        runs inference on a receipt image and returns extracted entities.

        args:
            image (PIL.Image): receipt image.
            words (List[str]): list of words extracted by tesseract from the receipt.
            boxes (List[List[int]]): bounding boxes for each word.

        Returns:
            dict: extracted entities from the receipt.
        """
        encoding = self.__processor(
            images=image,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )

        with torch.no_grad():
            outputs = self.__model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

        labels = [self.__model.config.id2label[p] for p in predictions]

        result = self._generate_output(words=words, labels=labels)

        return result