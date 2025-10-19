from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from ..config import get_settings
from ..models import Case
from ..schemas import ChatCitation, ChatResponse
from .embedding import EmbeddingClient
from .vectorstore import VectorStore


settings = get_settings()


class ChatService:
    def __init__(self, db: Session, vector_store: VectorStore, embedding_client: EmbeddingClient) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key and OpenAI else None

    def answer(
        self,
        question: str,
        categories: Optional[Sequence[str]] = None,
        themes: Optional[Sequence[str]] = None,
        orgs: Optional[Sequence[str]] = None,
        top_k: int = 6,
    ) -> ChatResponse:
        cases = self.db.scalars(select(Case)).all()
        filtered = [case for case in cases if self._passes_filters(case, categories, themes, orgs)]
        if not filtered:
            return ChatResponse(answer="該当する事例が見つかりませんでした。", citations=[])

        embedding = self.embedding_client.embed_texts([question])[0]
        candidate_ids = [case.id for case in filtered]
        matches = self.vector_store.search(embedding, top_k, filter_ids=candidate_ids)
        if not matches:
            return ChatResponse(answer="該当する事例が見つかりませんでした。", citations=[])

        ordered_cases = [self._find_case(filtered, case_id) for case_id, _ in matches]
        valid_cases = [case for case in ordered_cases if case]
        if not valid_cases:
            return ChatResponse(answer="該当する事例が見つかりませんでした。", citations=[])
        citations = [ChatCitation(id=case.id, title=case.title, link=case.link) for case in valid_cases]
        answer = self._generate_answer(question, valid_cases)
        return ChatResponse(answer=answer, citations=citations)

    def _generate_answer(self, question: str, cases: Iterable[Case]) -> str:
        valid_cases = [case for case in cases if case]
        if not valid_cases:
            return "該当する事例が見つかりませんでした。"
        if self._client is None:
            lines = []
            for idx, case in enumerate(valid_cases[:3], start=1):
                org = case.org or '組織名不明'
                lines.append(f"[{idx}] {case.title}（{org}）")
            summary = ' '.join(lines)
            return f"以下の事例が見つかりました。 {summary}"
        prompt = self._build_prompt(question, valid_cases)
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _build_prompt(self, question: str, cases: Sequence[Case]) -> List[dict]:
        context_lines = []
        for idx, case in enumerate(cases, start=1):
            themes = "、".join(case.themes_list)
            context_lines.append(
                f"[{idx}] タイトル: {case.title}\n組織: {case.org or '不明'}\nカテゴリ: {case.category or '不明'}\nテーマ: {themes or '不明'}\n概要: {case.summary}\nリンク: {case.link}"
            )
        context_text = "\n\n".join(context_lines)
        system_message = (
            "あなたは日本語で回答するアシスタントです。提供された事例情報から質問に簡潔に答え、"
            "参照した事例を [1][2] のように示してください。"
        )
        user_message = (
            f"質問: {question}\n\n利用可能な事例:\n{context_text}\n\n"
            "上記の事例のみを根拠に、2-3文で要約した回答を作り、文末に参照した番号を付けてください。"
        )
        return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    @staticmethod
    def _passes_filters(
        case: Case,
        categories: Optional[Sequence[str]],
        themes: Optional[Sequence[str]],
        orgs: Optional[Sequence[str]],
    ) -> bool:
        if categories and case.category not in set(categories):
            return False
        if orgs and case.org not in set(orgs):
            return False
        if themes:
            if not set(themes).intersection(case.themes_list):
                return False
        return True

    @staticmethod
    def _find_case(cases: Sequence[Case], case_id: str) -> Optional[Case]:
        for case in cases:
            if case.id == case_id:
                return case
        return None
