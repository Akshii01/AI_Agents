from typing import Literal

from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    story_name: str = Field(min_length=1)
    story: str = Field(min_length=1)


class SummaryResponse(BaseModel):
    key: str
    content: str


class RecipeChatRequest(BaseModel):
    message: str = Field(min_length=1)


class GeneralChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)


class RecipeChatResponse(BaseModel):
    reply: str


class GeneralChatResponse(BaseModel):
    reply: str
    tools_used: list[str]


class GeneratedRecipeResponse(BaseModel):
    reply: str
    memory_updates: list[dict[str, str]]
    memory_deletions: list[str]


class GeneratedGeneralResponse(BaseModel):
    reply: str
    session_summary: str
    tools_used: list[str]
    memory_updates: list[dict[str, str]]
    memory_deletions: list[str]


class SessionCreateRequest(BaseModel):
    title: str = Field(default="Story session", min_length=1, max_length=200)
    agent_type: Literal["story", "recipe", "general"] = "story"


class AuthRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, str]


class SessionSummaryResponse(SummaryResponse):
    id: str
    story_name: str
    created_at: str


class SessionResponse(BaseModel):
    id: str
    title: str
    agent_type: Literal["story", "recipe", "general"]
    created_at: str
    updated_at: str


class RecipeChatMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class SessionDetailResponse(SessionResponse):
    summaries: list[SessionSummaryResponse]
    recipe_chat_messages: list[RecipeChatMessageResponse]
    general_chat_messages: list[RecipeChatMessageResponse]
