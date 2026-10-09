import json
import os
import re
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq, GroqError
from pydantic import BaseModel, Field

from app.prompt_store import PROMPT_KEY, get_prompt, initialize_prompt_store
from app.session_store import (
    add_summary,
    create_session as store_create_session,
    get_session as store_get_session,
    initialize_session_store,
    list_sessions as store_list_sessions,
    session_exists,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class SummaryRequest(BaseModel):
    story_name: str = Field(min_length=1)
    story: str = Field(min_length=1)


class SummaryResponse(BaseModel):
    key: str
    content: str


class SessionCreateRequest(BaseModel):
    title: str = Field(default="Story session", min_length=1, max_length=200)


class SessionSummaryResponse(SummaryResponse):
    id: str
    story_name: str
    created_at: str


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class SessionDetailResponse(SessionResponse):
    summaries: list[SessionSummaryResponse]


def make_story_key(story_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", story_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    key = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    if not key:
        raise HTTPException(
            status_code=422, detail="story_name must contain at least one letter or number"
        )
    return key


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_prompt_store()
    initialize_session_store()
    yield


app = FastAPI(title="Story Summarizer Agent", lifespan=lifespan)

OUTPUT_CONTRACT = """Return exactly one valid JSON object with exactly two string fields:
{"key":"story-name","content":"formatted summary"}
Set key to the supplied story name in lowercase with words separated by hyphens.
Put the complete Markdown summary in content. Do not wrap the JSON in a code fence."""


def generate_summary(request: SummaryRequest) -> SummaryResponse:
    story_key = make_story_key(request.story_name)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    try:
        system_prompt = f"{get_prompt(PROMPT_KEY)}\n\n{OUTPUT_CONTRACT}"
    except (LookupError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Story name: {request.story_name}\n\nStory:\n{request.story}",
                },
            ],
            response_format={"type": "json_object"},
        )
    except GroqError as error:
        raise HTTPException(status_code=502, detail="Groq summarization request failed") from error

    if not completion.choices:
        raise HTTPException(status_code=502, detail="Groq returned no summary choices")

    response_content = completion.choices[0].message.content
    if not response_content:
        raise HTTPException(status_code=502, detail="Groq returned an empty summary")

    try:
        result = json.loads(response_content)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail="Groq returned invalid JSON") from error

    if (
        not isinstance(result, dict)
        or set(result) != {"key", "content"}
        or not isinstance(result["content"], str)
        or not result["content"].strip()
    ):
        raise HTTPException(status_code=502, detail="Groq returned an invalid summary format")

    return SummaryResponse(key=story_key, content=result["content"])


@app.post("/summarize", response_model=SummaryResponse)
def summarize_story(request: SummaryRequest) -> SummaryResponse:
    return generate_summary(request)


@app.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(request: SessionCreateRequest) -> SessionResponse:
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Session title cannot be blank")
    return SessionResponse(**store_create_session(title))


@app.get("/sessions", response_model=list[SessionResponse])
def list_sessions() -> list[SessionResponse]:
    return [SessionResponse(**session) for session in store_list_sessions()]


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def read_session(session_id: str) -> SessionDetailResponse:
    session = store_get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetailResponse(**session)


@app.post(
    "/sessions/{session_id}/summaries",
    response_model=SessionSummaryResponse,
    status_code=201,
)
def summarize_in_session(
    session_id: str, request: SummaryRequest
) -> SessionSummaryResponse:
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    summary = generate_summary(request)
    saved_summary = add_summary(
        session_id,
        request.story_name,
        summary.key,
        summary.content,
    )
    return SessionSummaryResponse(**saved_summary)
