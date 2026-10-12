from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    GeneralChatRequest,
    GeneralChatResponse,
    RecipeChatRequest,
    RecipeChatResponse,
    SummaryRequest,
    SummaryResponse,
)
from app.services.general_chat import generate_general_response
from app.services.recipe_chat import generate_recipe_response
from app.services.story import generate_summary
from app.storage.auth_store import get_current_user
from app.storage.session_store import (
    add_chat_exchange,
    add_general_chat_exchange,
    get_chat_history,
    get_general_chat_history,
    get_general_chat_session_summary,
    get_general_long_term_memories,
    get_long_term_memories,
    session_exists,
    update_general_long_term_memories,
    update_long_term_memories,
    update_session_title,
)

router = APIRouter()


def make_session_title(user_message: str) -> str:
    title = " ".join(user_message.strip().split())
    if len(title) > 80:
        title = f"{title[:77].rstrip()}..."
    return title or "New chat"


@router.post(
    "/sessions/{session_id}/recipe-chat",
    response_model=RecipeChatResponse,
)
def recipe_chat_in_session(
    session_id: str,
    request: RecipeChatRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> RecipeChatResponse:
    user_id = current_user["id"]
    if not session_exists(user_id, session_id, "recipe"):
        raise HTTPException(status_code=404, detail="Session not found")

    history = get_chat_history(user_id, session_id)
    generated = generate_recipe_response(
        request,
        history,
        get_long_term_memories(user_id),
    )
    if not history:
        update_session_title(user_id, session_id, make_session_title(request.message))
    update_long_term_memories(
        user_id,
        generated.memory_updates,
        generated.memory_deletions,
    )
    response = RecipeChatResponse(reply=generated.reply)
    add_chat_exchange(user_id, session_id, request.message.strip(), response.reply)
    return response


@router.post(
    "/sessions/{session_id}/general-chat",
    response_model=GeneralChatResponse,
)
def general_chat_in_session(
    session_id: str,
    request: GeneralChatRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> GeneralChatResponse:
    user_id = current_user["id"]
    if not session_exists(user_id, session_id, "general"):
        raise HTTPException(status_code=404, detail="Session not found")

    history = get_general_chat_history(user_id, session_id)
    generated = generate_general_response(
        request,
        history,
        get_general_long_term_memories(
            user_id,
            query=f"{request.message}\n{get_general_chat_session_summary(user_id, session_id)}",
        ),
        get_general_chat_session_summary(user_id, session_id),
    )
    if not history:
        update_session_title(
            user_id,
            session_id,
            make_session_title(request.message),
        )
    update_general_long_term_memories(
        user_id,
        generated.memory_updates,
        generated.memory_deletions,
    )
    add_general_chat_exchange(
        user_id,
        session_id,
        request.message.strip(),
        generated.reply,
        generated.session_summary,
    )
    return GeneralChatResponse(
        reply=generated.reply,
        tools_used=generated.tools_used,
    )


@router.post("/summarize", response_model=SummaryResponse)
def summarize_story(
    request: SummaryRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> SummaryResponse:
    return generate_summary(request)


@router.post("/recipe-chat", response_model=RecipeChatResponse)
def recipe_chat(
    request: RecipeChatRequest,
    current_user: dict[str, str] = Depends(get_current_user),
) -> RecipeChatResponse:
    user_id = current_user["id"]
    generated = generate_recipe_response(
        request,
        history=[],
        long_term_memories=get_long_term_memories(user_id),
    )
    update_long_term_memories(
        user_id,
        generated.memory_updates,
        generated.memory_deletions,
    )
    return RecipeChatResponse(reply=generated.reply)
