import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle
from typing import Generic, TypeVar


# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class your_feature(BaseModel):
     your_height: float
     your_weight: float

# 出力するデータ型の定義
class pokemon_data(BaseModel):
    id: float
    name: int
    type1: str
    type2: str
    height: float
    weight: float


# 学習済みのモデルの読み込み
pokemon_df = pd.read_csv("./pokemon_data2.csv")

# トップページ
@app.get('/')
def index():
    return {"Pokemon": 'pokemon_prediction'}

@app.get('/test')
def test():
    return {"test": your_feature["your_height"]}


# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/guess')
def make_predictions(features: your_feature):
    pokemon_df['差の総和'] = abs(pokemon_df['高さ'] - features.your_height) + abs(pokemon_df['重さ'] - features.your_weight)
    #pokemon_df['差の総和'] = abs(pokemon_df['高さ'] - your_feature["your_height"]) + abs(pokemon_df['重さ'] - your_feature["your_weight"])
    closest_pokemon = pokemon_df.loc[pokemon_df['差の総和'].idxmin()]

    return({'Pokemon':str(closest_pokemon['名前']),'pokemon_image':str(closest_pokemon['画像URL'])})