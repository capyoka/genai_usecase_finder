from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import List

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    org: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    themes: Mapped[str | None] = mapped_column(Text, nullable=True)
    link: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def themes_list(self) -> List[str]:
        if not self.themes:
            return []
        try:
            return json.loads(self.themes)
        except json.JSONDecodeError:
            return []

    @themes_list.setter
    def themes_list(self, values: List[str]) -> None:
        if values:
            self.themes = json.dumps(values, ensure_ascii=False)
        else:
            self.themes = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "org": self.org,
            "category": self.category,
            "themes": self.themes_list,
            "link": self.link,
        }
