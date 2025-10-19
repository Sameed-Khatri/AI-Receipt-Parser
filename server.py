from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from inference.routes import router
from inference.agent import Agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = Agent()
    app.state.agent = agent

    yield


app = FastAPI(title="Unikrew", lifespan=lifespan)
app.include_router(router=router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)