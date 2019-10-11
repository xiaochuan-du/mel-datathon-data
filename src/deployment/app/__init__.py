import os
from fastapi import (
    FastAPI, Depends, FastAPI, HTTPException
)
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from config import config_cls, basedir
from starlette.middleware.cors import CORSMiddleware


config = config_cls[os.getenv('APP_ENV', 'default')]


def _from_config(config_dict, obj):
    if isinstance(obj, str):
        # not applicable for right now
        # obj = import_string(obj)
        pass
    for key in dir(obj):
        if key.isupper():
            config_dict[key] = getattr(obj, key)

base_url = config.BASE_URL
app = FastAPI(
    title=config.PROJ_TITLE,
    description=config.PROJ_DESC,
    version=config.PROJ_VER,
    openapi_url=f"{base_url}/openapi.json",
    docs_url=f"{base_url}/docs",
)
app.mount(config.STATIC_URL, StaticFiles(
    directory=config.STATIC_DIR), name="static")

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http:localhost",
    "http:localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.middleware("http")
# async def db_session_middleware(request: Request, call_next):
#     from .models import SessionLocal
#     response = Response("Internal server error", status_code=500)
#     try:
#         request.state.db = SessionLocal()
#         response = await call_next(request)
#     finally:
#         request.state.db.close()
#     return response

from .routers import scoring_service
app.include_router(
    scoring_service.router,
    prefix="/score",
    tags=["scoring_service"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)