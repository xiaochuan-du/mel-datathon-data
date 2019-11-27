from typing import List
from fbprophet import Prophet
import os
import json
from fastapi import (
    Depends, FastAPI, HTTPException,
    APIRouter, Query, UploadFile, File
)
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import Response
from sqlalchemy.sql.expression import func
from pydantic import BaseModel, conint, confloat
from src.data.data_reformat import ROI_tifs
from src.models.predict_model import Predictor, tensor2score
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

#class HeatmapInfo(BaseModel):
 #   timestamp: conint(gt=0, lt=1e14) # invalid for earlier than 1970.01.01
  #  heatmap: List[List[float]]

class PlotInfo(BaseModel):
    heatmap: List[HeatmapInfo]
    lineplot: dict # [[x, y, val], ]

@router.post("/heatmap", response_model=List[HeatmapInfo])
async def get_heatmap(tile_query: TileQuery):
    """get_tiles for fe
    """
    #return json.load(
    #   open('data_demo.json', 'r'))
    tiles_info = ROI_tifs(json.dumps(tile_query.dict()))
    res = {}
    for ts in tiles_info['png_path']:
        img_path = tiles_info['png_path'][ts]
        img_path = img_path[img_path.find('results'):]
        res[ts] = f"{config.STATIC_DIR}/{img_path}"

    model = Predictor(res, im_mask=None)
    model.run(2)
    hm, line_res = model.gen_heatmaps()
    #json.dump(PlotInfo(
     #   heatmap=[
      #      HeatmapInfo(
       #         timestamp=info['timestamp'],
        #        heatmap=info['heatmap'].reshape(info['heatmap'].shape[0]*info['heatmap'].shape[1], 3).tolist()
         #   )
          #  for info in hm
       # ],
        #lineplot=line_res
    #).dict(), open('data_demo.json', 'w'))
    return  [
                HeatmapInfo(
                    timestamp=info['timestamp'],
                    heatmap=info['heatmap'].reshape(info['heatmap'].shape[0]*info['heatmap'].shape[1], 3).tolist()
                ) for info in hm
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
class Line(BaseModel):
    time_stamp: List[conint(gt=0, lt=1e14)]
    score: List[float]
class LineQuery(BaseModel):
    ds: List[conint(gt=0, lt=1e14)]
    y: List[float]


@router.post("/predict", response_model=Line)
async def get_predict_line(line_query: LineQuery):
    v_periods = 180
    df = pd.DataFrame({
            'ds': pd.to_datetime(np.array(line_query.dict()['ds']) * (10 ** 6)),
            'y': line_query.dict()['y']
            }).sort_values(by='ds')
    periods = (df.ds.max() - df.ds.min()).days + 1
    full_ds = pd.date_range(str(df.ds.min()), periods=periods, freq='D')
    new_df = pd.DataFrame({'ds': full_ds}).merge(df, how='left', on='ds')
    new_df['y'] = new_df.set_index('ds').y.interpolate(method='time').tolist()
    val = False
    if val:
        t_df, v_df = split(new_df, 0.9)
    else:
        t_df = new_df

    new_v_df = pd.DataFrame({
        'ds': pd.date_range(str(t_df.ds.max() + pd.Timedelta(1, 'D')), periods=v_periods, freq='D')
    })
    m = Prophet()
    m.fit(t_df)
    forecast = m.predict(new_v_df)
    scores, ups, downs = tensor2score(
            t_df.ds.tolist() + new_v_df.ds.tolist(),
            [np.array(x) for x in t_df.y.tolist() + forecast.yhat.tolist()])

    time_stamp = (scores.reset_index().date.astype('int')// 10 ** 6).tolist()
    print(time_stamp)    
    return Line(
        time_stamp=time_stamp,
        score=((scores.tgt_score.fillna(-1) * 100).astype('int32') / 100).tolist()
    )
