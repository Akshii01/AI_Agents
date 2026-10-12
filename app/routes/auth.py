from fastapi import APIRouter, Depends

from app.schemas import AuthRequest, AuthResponse
from app.storage.auth_store import get_current_user, login_user, register_user

router = APIRouter()


@router.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(request: AuthRequest) -> AuthResponse:
    user, token = register_user(request.email, request.password)
    return AuthResponse(access_token=token, user=user)


@router.post("/auth/login", response_model=AuthResponse)
def login(request: AuthRequest) -> AuthResponse:
    user, token = login_user(request.email, request.password)
    return AuthResponse(access_token=token, user=user)


@router.get("/auth/me")
def read_current_user(
    current_user: dict[str, str] = Depends(get_current_user),
) -> dict[str, str]:
    return current_user
