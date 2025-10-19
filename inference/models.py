from pydantic import BaseModel

class InferenceInput(BaseModel):
    image_path: str