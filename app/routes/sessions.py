import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    SessionCreateRequest,
    SessionDetailResponse,
    SessionResponse,
    SessionSummaryResponse,
    SummaryRequest,
)
from app.services.story import generate_summary
from app.storage.auth_store import get_current_user
from app.storage.session_store import (
    add_summary,
    create_session as store_create_session,
    get_session as store_get_session,
    list_sessions as store_list_sessions,
    session_exists,
    update_session_title,
)

router = APIRouter()
SERVER_INSTANCE_ID = str(uuid.uuid4())


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(
    request: SessionCreateRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> SessionResponse:
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Session title cannot be blank")
    return SessionResponse(
        **store_create_session(
            current_user["id"],
            title,
            request.agent_type,
        )
    )


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    agent_type: Literal["story", "recipe", "general"] | None = None,
    current_user: dict[str, str] = Depends(get_current_user),
) -> list[SessionResponse]:
    return [
        SessionResponse(**session)
        for session in store_list_sessions(current_user["id"], agent_type)
    ]


@router.get("/server-instance")
def get_server_instance() -> dict[str, str]:
    return {"id": SERVER_INSTANCE_ID}


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def read_session(
    session_id: str,
    current_user: dict[str, str] = Depends(get_current_user),
) -> SessionDetailResponse:
    session = store_get_session(current_user["id"], session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetailResponse(**session)


@router.post(
    "/sessions/{session_id}/summaries",
    response_model=SessionSummaryResponse,
    status_code=201,
)
def summarize_in_session(
    session_id: str,
    request: SummaryRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> SessionSummaryResponse:
    user_id = current_user["id"]
    if not session_exists(user_id, session_id, "story"):
        raise HTTPException(status_code=404, detail="Session not found")
    summary = generate_summary(request)
    session_before = store_get_session(user_id, session_id)
    saved_summary = add_summary(
        user_id,
        session_id,
        request.story_name,
        summary.key,
        summary.content,
    )
    if session_before is not None and not session_before["summaries"]:
        update_session_title(user_id, session_id, request.story_name.strip()[:80])
    return SessionSummaryResponse(**saved_summary)
