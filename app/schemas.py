from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CaseItem(BaseModel):
    id: str
    title: str
    summary: str
    org: Optional[str] = None
    category: Optional[str] = None
    themes: List[str] = Field(default_factory=list)
    link: str
    score: Optional[float] = None


class CaseSearchResponse(BaseModel):
    total: int
    items: List[CaseItem]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatFilters(BaseModel):
    category: List[str] = Field(default_factory=list)
    theme: List[str] = Field(default_factory=list)
    org: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    filters: Optional[ChatFilters] = None
    top_k: Optional[int] = None


class ChatCitation(BaseModel):
    id: str
    title: str
    link: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[ChatCitation]


class ImportPreviewRow(BaseModel):
    category: str | None
    org: str | None
    title: str
    summary: str
    themes: List[str]
    link: str


class ImportPreviewResponse(BaseModel):
    rows: List[ImportPreviewRow]


class ImportResult(BaseModel):
    total_rows: int
    imported: int
    skipped: int
