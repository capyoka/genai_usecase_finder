from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...config import get_settings
from ...schemas import ChatRequest, ChatResponse
from ...services.chat import ChatService
from ..deps import get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])
settings = get_settings()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    question = request.messages[-1].content
    filters = request.filters or None
    top_k = request.top_k or settings.chat_search_top_k
    return service.answer(
        question,
        categories=filters.category if filters else None,
        themes=filters.theme if filters else None,
        orgs=filters.org if filters else None,
        top_k=top_k,
    )
