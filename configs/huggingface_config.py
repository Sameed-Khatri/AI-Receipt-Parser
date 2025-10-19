import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


class Hunggingface:
    def __init__(self):
        self.__model_id = "Sameed1/smdk-layoutlmv3-receipts"
        self.__processor = LayoutLMv3Processor.from_pretrained(self.__model_id)
        self.__model = LayoutLMv3ForTokenClassification.from_pretrained(self.__model_id)
        self.__model.to("cpu")
        self.__model.eval()
    

    def _generate_output(self, words, labels):
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


    async def run_inference(self, image, words, boxes):
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