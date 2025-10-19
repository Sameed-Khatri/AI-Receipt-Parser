import os, json
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict
from pydantic import BaseModel


load_dotenv()

class LLM:
    def __init__(self):
        self.__llm_model = "openai/gpt-oss-120b"
        self.__temperature = 0.3
        self.__api_key = os.getenv("GROQ_API_KEY")
        self.__client = Groq(
            api_key=self.__api_key
        )


    async def chat(self, messages: List[Dict], pydantic_model: BaseModel, response_format: Dict = {}):
        response = self.__client.chat.completions.create(
            model=self.__llm_model,
            temperature=self.__temperature,
            messages=messages,
            response_format=response_format
        )

        raw_result = json.loads(response.choices[0].message.content or "{}")
        result = pydantic_model.model_validate(raw_result)

        return result.model_dump_json(indent=2)