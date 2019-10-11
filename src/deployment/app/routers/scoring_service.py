from typing import List
import json
from fastapi import (
    Depends, FastAPI, HTTPException,
    APIRouter, Query, UploadFile, File
)
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import Response
from sqlalchemy.sql.expression import func
from pydantic import BaseModel, conint, confloat

router = APIRouter()

class ImageScore(BaseModel):
    timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    score: confloat(gt=.0, lt=1.)

class ScoreInfo(BaseModel):
    st_timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    end_timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01

# def normalize_dict(**kwargs):
#     output = {}
#     for key in kwargs:
#         if isinstance(kwargs[key], str):
#             output[key] = kwargs[key]
#         elif isinstance(kwargs[key], dict):
#             output[key] = json.kwargs[key]
#     return {
#         key: 
#             kwargs[key] if isinstance(kwargs[key], str) elif isinstance(kwargs[key], dict)
#         for key in kwargs
#     }

# , file: UploadFile = File(...)
#     content = await file.read()
# with open(file.filename, 'wb') as file_obj:
#     file_obj.write(content)

from starlette.responses import FileResponse
import tempfile

@router.post("/mock", response_model=List[ImageScore])
async def mock_satdb(score_info: ScoreInfo):
    return FileResponse('/Users/kevindu/Downloads/111.jpeg', media_type="image/png")

import httpx
client = httpx.AsyncClient()

@router.post("/predict", response_model=List[ImageScore])
async def predict_scores(score_info: ScoreInfo, coordinates: List[float] = []):
    r = await client.post('http://0.0.0.0:5000/score/mock', data=json.dumps(score_info.dict()))
    with open('tmp.jpeg', 'wb') as file_obj:
        file_obj.write(r.content)
    score = ImageScore(timestamp=1, score=0.5)
    return [score.dict()]