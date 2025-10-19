from __future__ import annotations

import csv
import io
import re
import uuid
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Case
from ..schemas import CaseItem, CaseSearchResponse, ImportResult
from .embedding import EmbeddingClient
from .vectorstore import VectorStore


settings = get_settings()
THEME_SPLIT_PATTERN = re.compile(r"[、,，/\\|｜・]")


class CaseService:
    def __init__(self, db: Session, vector_store: VectorStore, embedding_client: EmbeddingClient) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embedding_client = embedding_client

    # Data utilities
    def get_filter_values(self) -> Tuple[List[str], List[str], List[str]]:
        cases = self.db.scalars(select(Case)).all()
        categories = sorted({case.category for case in cases if case.category})
        themes = sorted({theme for case in cases for theme in case.themes_list})
        orgs = sorted({case.org for case in cases if case.org})
        return categories, themes, orgs

    def _normalize_row(self, row: dict) -> Optional[Case]:
        title = (row.get("タイトル") or "").strip()
        summary = (row.get("概要") or "").strip()
        link = self._normalize_link(row.get("リンク"))
        if not title or not summary or not link:
            return None
        category = (row.get("カテゴリ") or "").strip() or None
        org = (row.get("組織名") or "").strip() or None
        raw_themes = row.get("テーマ") or ""
        themes = [t.strip() for t in THEME_SPLIT_PATTERN.split(raw_themes) if t.strip()]
        case = Case(
            title=title,
            summary=summary,
            link=link,
            category=category,
            org=org,
        )
        case.themes_list = themes
        return case

    def _normalize_link(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        link = value.strip()
        if not link:
            return None
        parsed = urlparse(link)
        if not parsed.scheme:
            link = "https://" + link
        return link

    def parse_csv(self, file_bytes: bytes) -> Tuple[List[Case], int, int]:
        text = file_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        cases: List[Case] = []
        total_rows = 0
        skipped = 0
        for row in reader:
            total_rows += 1
            normalized = self._normalize_row(row)
            if normalized is None:
                skipped += 1
                continue
            cases.append(normalized)
        return cases, total_rows, skipped

    def import_cases(self, cases: Sequence[Case]) -> ImportResult:
        total = len(cases)
        imported = 0
        skipped = 0
        if not cases:
            return ImportResult(total_rows=0, imported=0, skipped=0)

        embeddings_inputs: List[Tuple[str, str]] = []
        to_remove: List[str] = []

        for case in cases:
            existing = self.db.scalar(select(Case).where(Case.link == case.link))
            if existing:
                to_remove.append(existing.id)
                existing.title = case.title
                existing.summary = case.summary
                existing.category = case.category
                existing.org = case.org
                existing.themes_list = case.themes_list
                case_id = existing.id
            else:
                case_id = str(uuid.uuid4())
                case.id = case_id
                self.db.add(case)
            embeddings_inputs.append((case_id, f"{case.title}\n{case.summary}"))
            imported += 1

        self.db.commit()

        if to_remove:
            self.vector_store.remove(to_remove)

        vectors = self.embedding_client.embed_texts(text for _, text in embeddings_inputs)
        if vectors:
            self.vector_store.add([case_id for case_id, _ in embeddings_inputs], vectors)

        return ImportResult(total_rows=total, imported=imported, skipped=skipped)

    def search_cases(
        self,
        q: Optional[str],
        categories: Optional[Sequence[str]],
        themes: Optional[Sequence[str]],
        orgs: Optional[Sequence[str]],
        page: int,
        size: int,
    ) -> CaseSearchResponse:
        all_cases = self.db.scalars(select(Case)).all()
        filtered = self._filter_cases(all_cases, categories, themes, orgs)
        if q:
            embedding = self.embedding_client.embed_texts([q])[0]
            candidate_ids = [case.id for case in filtered]
            ranked = self.vector_store.search(embedding, settings.case_search_top_k, filter_ids=candidate_ids)
            case_map = {case.id: case for case in filtered}
            ordered_cases = [
                self._build_case_item(case_map[case_id], score)
                for case_id, score in ranked
                if case_id in case_map
            ]
        else:
            ordered_cases = [self._build_case_item(case, None) for case in filtered]
            ordered_cases.sort(key=lambda item: (item.title.lower()))

        total = len(ordered_cases)
        start = (page - 1) * size
        end = start + size
        paged_items = ordered_cases[start:end]
        return CaseSearchResponse(total=total, items=paged_items)

    def _filter_cases(
        self,
        cases: Iterable[Case],
        categories: Optional[Sequence[str]],
        themes: Optional[Sequence[str]],
        orgs: Optional[Sequence[str]],
    ) -> List[Case]:
        filtered = []
        categories_set = {c for c in categories or []}
        theme_set = {t for t in themes or []}
        org_set = {o for o in orgs or []}
        for case in cases:
            if categories_set and (case.category not in categories_set):
                continue
            if org_set and (case.org not in org_set):
                continue
            if theme_set and not (theme_set.intersection(case.themes_list)):
                continue
            filtered.append(case)
        return filtered

    def _build_case_item(self, case: Case, score: Optional[float]) -> CaseItem:
        return CaseItem(
            id=case.id,
            title=case.title,
            summary=case.summary,
            org=case.org,
            category=case.category,
            themes=case.themes_list,
            link=case.link,
            score=score,
        )


def limit_page_size(size: int) -> int:
    if size <= 0:
        return settings.default_page_size
    return min(size, settings.max_page_size)
