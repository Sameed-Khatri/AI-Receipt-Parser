import os
from dotenv import load_dotenv
from configs.groq_config import LLM
from configs.tesseract_config import OCR
from configs.huggingface_config import Hunggingface
from .models import AgentOutput


load_dotenv()



class Agent:
    """
    agent class for orchestrating the tesseract ocr, fine tuned layoutlmv3 vision model, and gorq llm pipeline for receipt parsing.

    attributes:
        __structured_output: pydantic model for agent output.
        __llm: groq llm wrapper instance.
        __ocr: tesseract ocr wrapper instance.
        __hunggingface_model: huggingface layoutlmv3 wrapper instance.
        __system_prompt: system prompt string for llm.
        __human_prompt: human prompt string for llm.
    """

    def __init__(self):
        """
        initializes the agent, loads prompts, and sets up model instances.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(base_dir, "prompts")
        system_prompt_path = os.path.join(prompts_dir, "system_prompt.txt")
        human_prompt_path = os.path.join(prompts_dir, "human_prompt.txt")

        self.__structured_output = AgentOutput
        self.__llm = LLM()
        self.__ocr = OCR()
        self.__hunggingface_model = Hunggingface()

        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.__system_prompt = f.read()

        with open(human_prompt_path, "r", encoding="utf-8") as f:
            self.__human_prompt = f.read()


    async def pipeline(self, image_path: str):
        """
        runs the full pipeline: tesseract, layoutlmv3 inference, prompt formatting, and llm reasoning.

        args:
            image_path (str): path to the receipt image.

        returns:
            str: json string of validated agent output.
        """
        ocr_output = await self.__ocr.run_ocr(image_path=image_path)
        words, boxes, image = ocr_output["words"], ocr_output["boxes"], ocr_output["image"]

        model_inference_result = await self.__hunggingface_model.run_inference(image=image, words=words, boxes=boxes)
        ocr_text = " ".join(words)

        formatted_human_prompt = self.__human_prompt.format(
            ocr_text=ocr_text,
            extracted_json=model_inference_result
        )

        messages = [
            {
                "role": "system",
                "content": self.__system_prompt
            },
            {
                "role": "user",
                "content": formatted_human_prompt
            }
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_output",
                "schema": self.__structured_output.model_json_schema()
            }
        }

        result = await self.__llm.chat(messages=messages, response_format=response_format, pydantic_model=self.__structured_output)

        return result
