from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from ...schemas import CaseSearchResponse
from ...services.cases import CaseService, limit_page_size
from ..deps import get_case_service

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("", response_model=CaseSearchResponse)
def search_cases(
    q: Optional[str] = None,
    category: Optional[List[str]] = Query(default=None, alias="category[]"),
    theme: Optional[List[str]] = Query(default=None, alias="theme[]"),
    org: Optional[List[str]] = Query(default=None, alias="org[]"),
    page: int = 1,
    size: int = 20,
    service: CaseService = Depends(get_case_service),
) -> CaseSearchResponse:
    size = limit_page_size(size)
    page = max(page, 1)
    return service.search_cases(q, category, theme, org, page, size)


@router.get('/filters')
def get_filters(service: CaseService = Depends(get_case_service)):
    categories, themes, orgs = service.get_filter_values()
    return {'categories': categories, 'themes': themes, 'orgs': orgs}
