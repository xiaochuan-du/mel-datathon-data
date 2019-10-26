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
from src.data.data_reformat import ROI_tifs

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

# @router.post("/mock", response_model=List[ImageScore])
# async def mock_satdb(score_info: ScoreInfo):
#     return FileResponse('/Users/kevindu/Downloads/111.jpeg', media_type="image/png")

# import httpx
# client = httpx.AsyncClient()

# @router.post("/predict", response_model=List[ImageScore])
# async def predict_scores(score_info: ScoreInfo, coordinates: List[float] = []):
#     r = await client.post('http://0.0.0.0:5000/score/mock', data=json.dumps(score_info.dict()))
#     with open('tmp.jpeg', 'wb') as file_obj:
#         file_obj.write(r.content)
#     score = ImageScore(timestamp=1, score=0.5)
#     return [score.dict()]

# @router.post("/image/predict", response_model=List[ImageScore])
# async def predict_scores(file: UploadFile = File(...)):
#     r = await client.post('http://0.0.0.0:5000/score/mock', data=json.dumps(score_info.dict()))
#     with open('tmp.jpeg', 'wb') as file_obj:
#         file_obj.write(r.content)
#     score = ImageScore(timestamp=1, score=0.5)
#     return [score.dict()]

class TileInfo(BaseModel):
    timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    file_uri: str

class TileQuery(BaseModel):
    geo_json: dict

@router.post("/tiles", response_model=List[TileInfo])
async def get_tiles(tile_query: TileQuery):
    """get_tiles for fe
        ROI_large = {"type": "Feature","geometry": {"type": "Polygon",
    "coordinates": [[148.60709030303114, -20.540043246963264],
      [148.69607543743531, -20.539590412428996],
      [148.6865658493269, -20.595756032466892],
      [148.6275658455197,-20.606209452942387]]},"properties": {"name": ""}}
    ROI_one = {"type": "Feature","geometry": {"type": "Polygon",
        "coordinates": [(148.66, -20.55),
    (148.68, -20.55),
    (148.68, -20.57),
    (148.66, -20.57)]},"properties": {"name": ""}}
    """    
    tiles_info = ROI_tifs(tile_query.geo_json)
    return [
        TileInfo(timestamp=ts, file_uri=tiles_info['png_path'][ts])
        for ts in tiles_info['png_path']
    ]

class HeatmapInfo(BaseModel):
    timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    heatmap: List[List[float]] # [[x, y, val], ]


DEMO_DATA = [[0,0,5],[0,1,1],[0,2,0],[0,3,0],[0,4,0],[0,5,0],[0,6,0],[0,7,0],[0,8,0],[0,9,0],[0,10,0],[0,11,2],[0,12,4],[0,13,1],[0,14,1],[0,15,3],[0,16,4],[0,17,6],[0,18,4],[0,19,4],[0,20,3],[0,21,3],[0,22,2],[0,23,5],[1,0,7],[1,1,0],[1,2,0],[1,3,0],[1,4,0],[1,5,0],[1,6,0],[1,7,0],[1,8,0],[1,9,0],[1,10,5],[1,11,2],[1,12,2],[1,13,6],[1,14,9],[1,15,11],[1,16,6],[1,17,7],[1,18,8],[1,19,12],[1,20,5],[1,21,5],[1,22,7],[1,23,2],[2,0,1],[2,1,1],[2,2,0],[2,3,0],[2,4,0],[2,5,0],[2,6,0],[2,7,0],[2,8,0],[2,9,0],[2,10,3],[2,11,2],[2,12,1],[2,13,9],[2,14,8],[2,15,10],[2,16,6],[2,17,5],[2,18,5],[2,19,5],[2,20,7],[2,21,4],[2,22,2],[2,23,4],[3,0,7],[3,1,3],[3,2,0],[3,3,0],[3,4,0],[3,5,0],[3,6,0],[3,7,0],[3,8,1],[3,9,0],[3,10,5],[3,11,4],[3,12,7],[3,13,14],[3,14,13],[3,15,12],[3,16,9],[3,17,5],[3,18,5],[3,19,10],[3,20,6],[3,21,4],[3,22,4],[3,23,1],[4,0,1],[4,1,3],[4,2,0],[4,3,0],[4,4,0],[4,5,1],[4,6,0],[4,7,0],[4,8,0],[4,9,2],[4,10,4],[4,11,4],[4,12,2],[4,13,4],[4,14,4],[4,15,14],[4,16,12],[4,17,1],[4,18,8],[4,19,5],[4,20,3],[4,21,7],[4,22,3],[4,23,0],[5,0,2],[5,1,1],[5,2,0],[5,3,3],[5,4,0],[5,5,0],[5,6,0],[5,7,0],[5,8,2],[5,9,0],[5,10,4],[5,11,1],[5,12,5],[5,13,10],[5,14,5],[5,15,7],[5,16,11],[5,17,6],[5,18,0],[5,19,5],[5,20,3],[5,21,4],[5,22,2],[5,23,0],[6,0,1],[6,1,0],[6,2,0],[6,3,0],[6,4,0],[6,5,0],[6,6,0],[6,7,0],[6,8,0],[6,9,0],[6,10,1],[6,11,0],[6,12,2],[6,13,1],[6,14,3],[6,15,4],[6,16,0],[6,17,0],[6,18,0],[6,19,0],[6,20,1],[6,21,2],[6,22,2],[6,23,6]];


@router.post("/heatmap", response_model=List[HeatmapInfo])
async def get_heatmap(tile_query: TileQuery):
    """get_tiles for fe
    """
    return [
        HeatmapInfo(timestamp=1571546212832, heatmap=DEMO_DATA),
        HeatmapInfo(timestamp=1571542212832, heatmap=DEMO_DATA),
        HeatmapInfo(timestamp=15715468212832, heatmap=DEMO_DATA)
    ]

class ChartPoint(BaseModel):
    timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    y: List[float] # [[x, y, val], ]

@router.post("/line_chart", response_model=List[HeatmapInfo])
async def get_line_chart(tile_query: TileQuery):
    """get_tiles for fe
    """
    return [
        HeatmapInfo(timestamp=1571546212832, heatmap=DEMO_DATA),
        HeatmapInfo(timestamp=1571542212832, heatmap=DEMO_DATA),
        HeatmapInfo(timestamp=15715468212832, heatmap=DEMO_DATA)
    ]