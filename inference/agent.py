import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from configs.groq_config import LLM
from configs.tesseract_config import OCR
from configs.huggingface_config import Hunggingface


load_dotenv()


class AgentOutput(BaseModel):
    company: str = Field(..., description="Name of the company")
    date: str = Field(..., description="Date of the receipt")
    address: str = Field(..., description="Address of the company")
    total: str = Field(..., description="Total amount on the receipt with currency if mentioned")
    agent_comment: str = Field(..., description="Summary of action performed by the llm and data found in ocr and vision model json")


class Agent:
    def __init__(self):
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
