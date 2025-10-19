from __future__ import annotations

from typing import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..services.cases import CaseService
from ..services.chat import ChatService
from ..services.embedding import EmbeddingClient
from ..services.vectorstore import VectorStore


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_embedding_client(request: Request) -> EmbeddingClient:
    return request.app.state.embedding_client


def get_case_service(
    db: Session = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
) -> CaseService:
    return CaseService(db, vector_store, embedding_client)


def get_chat_service(
    db: Session = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
) -> ChatService:
    return ChatService(db, vector_store, embedding_client)
