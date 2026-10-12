from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.sessions import router as sessions_router
from app.storage.auth_store import initialize_auth_store
from app.storage.prompt_store import initialize_prompt_store
from app.storage.session_store import initialize_session_store

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_auth_store()
    initialize_prompt_store()
    initialize_session_store()
    yield


app = FastAPI(title="Agent", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(sessions_router)
