from fastapi import APIRouter, Request
from .models import InferenceInput


router = APIRouter(prefix="/unikrew")

@router.post(path="/inference")
async def inference_handler(payload: InferenceInput, request: Request):
    """
    endpoint to run the inference pipeline on the provided receipt image path.

    args:
        payload (InferenceInput): structured request input containing image path.
        request (Request): fastapi request object containing app state with the intialized agent object.

    returns:
        dict: agent response containing extracted receipt details with agent comment.
    """
    agent = request.app.state.agent
    response = await agent.pipeline(image_path=payload.image_path)
    return response