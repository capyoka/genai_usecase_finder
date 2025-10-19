from __future__ import annotations

from typing import Union

from fastapi import APIRouter, Depends, File, UploadFile

from ...schemas import ImportPreviewResponse, ImportResult
from ...services.cases import CaseService
from ..deps import get_case_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/import", response_model=ImportResult | ImportPreviewResponse)
async def import_cases(
    file: UploadFile = File(...),
    service: CaseService = Depends(get_case_service),
    preview: bool = False,
) -> ImportResult | ImportPreviewResponse:
    data = await file.read()
    cases, total_rows, skipped_rows = service.parse_csv(data)
    if preview:
        preview_rows = [
            {
                "category": case.category,
                "org": case.org,
                "title": case.title,
                "summary": case.summary,
                "themes": case.themes_list,
                "link": case.link,
            }
            for case in cases[:100]
        ]
        return ImportPreviewResponse(rows=preview_rows)
    result = service.import_cases(cases)
    return ImportResult(total_rows=total_rows, imported=result.imported, skipped=skipped_rows + result.skipped)
