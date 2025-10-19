from fastapi import APIRouter, Request
from .models import InferenceInput


router = APIRouter(prefix="/unikrew")

@router.post(path="/inference")
async def inference_handler(payload: InferenceInput, request: Request):
    agent = request.app.state.agent
    response = await agent.pipeline(image_path=payload.image_path)
    return response