from typing import List
import os
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
from src.models.predict_model import Predictor
from config import config_cls, basedir


config = config_cls[os.getenv('APP_ENV', 'default')]

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

from starlette.responses import FileResponse
import tempfile


class TileInfo(BaseModel):
    timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    file_uri: str

class TileQuery(BaseModel):
    type: str = "Feature"
    geometry: dict = {"type": "Polygon",
        "coordinates": [[148.60709030303114, -20.540043246963264],
        [148.69607543743531, -20.539590412428996],
        [148.6865658493269, -20.595756032466892],
        [148.6275658455197,-20.606209452942387]]}
    properties: dict = {"name": ""}    


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
    tiles_info = ROI_tifs(json.dumps(tile_query.dict()))
    res = []
    for ts in tiles_info['png_path']:
        img_path = tiles_info['png_path'][ts]
        img_path = img_path[img_path.find('results'):]
        res.append(TileInfo(timestamp=ts, file_uri=f"{config.STATIC_URL}/{img_path}"))
    return res

class HeatmapInfo(BaseModel):
    timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
    heatmap: List[List[float]] # [[x, y, val], ]


@router.post("/heatmap", response_model=List[HeatmapInfo])
async def get_heatmap(tile_query: TileQuery):
    """get_tiles for fe
    """
    tiles_info = ROI_tifs(json.dumps(tile_query.dict()))
    res = {}
    for ts in tiles_info['png_path']:
        img_path = tiles_info['png_path'][ts]
        img_path = img_path[img_path.find('results'):]
        res[ts] = f"{config.STATIC_DIR}/{img_path}"

    model = Predictor(res, im_mask=None)
    model.run(2)
    hm = model.gen_heatmaps()
    return [
        HeatmapInfo(
            timestamp=info['timestamp'],
            heatmap=info['heatmap'].reshape(info['heatmap'].shape[0]*info['heatmap'].shape[1], 3).tolist()
        )
        for info in hm
    ]

# class ChartPoint(BaseModel):
#     timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
#     y: List[float] # [[x, y, val], ]

# @router.post("/line_chart", response_model=List[HeatmapInfo])
# async def get_line_chart(tile_query: TileQuery):
#     """get_tiles for fe
#     """
#     return [
#         HeatmapInfo(timestamp=1571546212832, heatmap=DEMO_DATA),
#         HeatmapInfo(timestamp=1571542212832, heatmap=DEMO_DATA),
#         HeatmapInfo(timestamp=15715468212832, heatmap=DEMO_DATA)
#     ]