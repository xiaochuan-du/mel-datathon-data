from fastapi import FastAPI
from pydantic import BaseModel, Schema
from typing import List
from run import config
from src.models.predict_model import load_infer_obj
from src.models.predict_model import predict


learner = load_infer_obj(
    **config.ML_PARMS
)

class GenResponse(BaseModel):
    data: dict
    msg: str = ''
    status: str

class Item(BaseModel):
    data: List[float] = Schema(
        ..., min_length=1, max_length=1e4,
        description="The signal size must be greater than one and less than 1e4"
    )
    sample_r: int = Schema(
        ..., gt=0, lt=1e4,
        description="The sample rate must be greater than zero and less than 1e4")

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=('GET', 'POST', 'OPTIONS'))


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/item/predicton", response_model=GenResponse)
async def predict_item(item: Item):
    resp = predict(learner, item)
    return GenResponse(**resp)