from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api.routes import admin, cases, chat
from .api.deps import get_case_service
from .config import get_settings
from .database import Base, engine
from .services.cases import CaseService
from .services.embedding import EmbeddingClient
from .services.vectorstore import VectorStore

settings = get_settings()
app = FastAPI(title=settings.app_name)

Base.metadata.create_all(bind=engine)

app.state.embedding_client = EmbeddingClient()
app.state.vector_store = VectorStore(dimension=app.state.embedding_client.dimension)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(cases.router)
api_router.include_router(chat.router)
api_router.include_router(admin.router)
app.include_router(api_router)


@app.get("/", response_class=RedirectResponse)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/cases", status_code=302)


@app.get("/cases", response_class=HTMLResponse)
async def cases_page(
    request: Request,
    service: CaseService = Depends(get_case_service),
) -> HTMLResponse:
    categories, themes, orgs = service.get_filter_values()
    context = {
        "request": request,
        "title": "事例検索",
        "categories": categories,
        "themes": themes,
        "orgs": orgs,
        "page_size": settings.default_page_size,
    }
    return templates.TemplateResponse("cases.html", context)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    service: CaseService = Depends(get_case_service),
) -> HTMLResponse:
    categories, themes, orgs = service.get_filter_values()
    context = {
        "request": request,
        "title": "チャット検索",
        "categories": categories,
        "themes": themes,
        "orgs": orgs,
    }
    return templates.TemplateResponse("chat.html", context)


@app.get("/admin/import", response_class=HTMLResponse)
async def admin_import_page(request: Request) -> HTMLResponse:
    context = {
        "request": request,
        "title": "CSV取り込み",
    }
    return templates.TemplateResponse("admin_import.html", context)
