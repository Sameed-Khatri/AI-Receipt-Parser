from pydantic import BaseModel, Field



class AgentOutput(BaseModel):
    company: str = Field(..., description="Name of the company")
    date: str = Field(..., description="Date of the receipt")
    address: str = Field(..., description="Address of the company")
    total: str = Field(..., description="Total amount on the receipt with currency if mentioned")
    agent_comment: str = Field(..., description="Summary of action performed by the llm and data found in ocr and vision model json")


class InferenceInput(BaseModel):
    image_path: str