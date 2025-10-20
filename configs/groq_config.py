import os, json
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict
from pydantic import BaseModel


load_dotenv()

class LLM:
    """
    wrapper class for interacting with the groq api.

    attributes:
        __llm_model (str): the model name to use for inference.
        __temperature (float): sampling temperature for the model.
        __api_key (str): api key loaded from environment variables.
        __client (groq): groq client instance for api calls.
    """

    def __init__(self):
        """
        initializes the llm class with model parameters and api key.
        """
        self.__llm_model = "openai/gpt-oss-120b"
        self.__temperature = 0.3
        self.__api_key = os.getenv("GROQ_API_KEY")
        self.__client = Groq(
            api_key=self.__api_key
        )


    async def chat(self, messages: List[Dict], pydantic_model: BaseModel, response_format: Dict = {}):
        """
        sends a chat completion request to the gorq API and validates the response.

        args:
            messages (List[Dict]): list of message dicts for the llm conversation.
            pydantic_model (BaseModel): pydantic model for response validation.
            response_format (Dict, optional): structured response format for the llm.

        returns:
            str: json string of the validated response.
        """
        response = self.__client.chat.completions.create(
            model=self.__llm_model,
            temperature=self.__temperature,
            messages=messages,
            response_format=response_format
        )

        raw_result = json.loads(response.choices[0].message.content or "{}")
        result = pydantic_model.model_validate(raw_result)

        return result.model_dump_json(indent=2)