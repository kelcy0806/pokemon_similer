import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle
from typing import Generic, TypeVar
import random

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class your_feature(BaseModel):
     your_height: float
     your_weight: float
     HP:float
     こうげき:float
     ぼうぎょ:float
     とくこう:float
     とくぼう:float
     すばやさ:float

# 出力するデータ型の定義
class pokemon_data(BaseModel):
    id: float
    name: int
    type1: str
    type2: str
    height: float
    weight: float


# データのの読み込み
pokemon_df = pd.read_csv("./pokemon_data2.csv")

# 学習済みのモデルの読み込み
model = pickle.load(open('models/Pokemon_model', 'rb'))

# トップページ
@app.get('/')
def index():
    return {"Pokemon": 'pokemon_prediction2'}

@app.get('/test')
def test():
    return {"test": 'your_feature["your_height"]'}


# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/guess')
def make_predictions(features:your_feature):

    pokemon_df['差の総和'] = abs(pokemon_df['高さ'] - features.your_height)*100 + abs(pokemon_df['重さ'] - features.your_weight)
    #pokemon_df['差の総和'] = abs(pokemon_df['高さ'] - features['your_height']) + abs(pokemon_df['重さ'] - features['your_weight'])
    closest_pokemon = pokemon_df.loc[pokemon_df['差の総和'].idxmin()]

    #ランダムに種族値割り振り
    # 合計値
    target_sum = 599

    # 乱数のシードを設定して再現性を確保
    #np.random.seed(40)

    # 各要素にランダムな整数を割り振る
    random_values = {key: np.random.randint(50, 120) for key in ['HP','こうげき','ぼうぎょ','とくこう','とくぼう','すばやさ']}

    random_values['HP'] = random_values['HP']+int(features.ぼうぎょ*random_values['HP']/2)
    random_values['こうげき'] = random_values['こうげき']+int(features.HP*random_values['こうげき']/2)-int(features.こうげき*random_values['こうげき']/2)
    random_values['ぼうぎょ'] = random_values['ぼうぎょ']+int(features.HP*random_values['ぼうぎょ']/2)+int(features.こうげき*random_values['ぼうぎょ']/2)
    random_values['とくこう'] = random_values['とくこう']-int(features.HP*random_values['とくこう']/2)-int(features.こうげき*random_values['とくこう']/2)
    random_values['とくぼう'] = random_values['とくぼう']-int(features.HP*random_values['とくぼう']/2)+int(features.こうげき*random_values['ぼうぎょ']/2)
    random_values['すばやさ'] = random_values['すばやさ']+int(features.ぼうぎょ*random_values['すばやさ']/2)

    # 合計が目標値になるように調整
    while sum(random_values.values()) >= target_sum:
        # 合計が目標値より大きい場合は再度ランダムな値を割り振る
        if sum(random_values.values()) > target_sum:
            key_to_adjust = random.choice(['HP','こうげき','ぼうぎょ','とくこう','とくぼう','すばやさ'])
            random_values[key_to_adjust] -= 1
        # 合計が目標値より小さい場合は再度ランダムな値を割り振る
        else:
            key_to_adjust = random.choice(['HP','こうげき','ぼうぎょ','とくこう','とくぼう','すばやさ'])
            random_values[key_to_adjust] += 1

    #random_values_list = np.array(random_values.values(),np.float32)
    #a = {'prediction':model.predict(random_values_list)}

    random_values['合計'] = int(random_values['HP']+random_values['こうげき']+random_values['ぼうぎょ']+random_values['とくこう']+random_values['とくぼう']+random_values['すばやさ'])
    random_values.update(prediction=model.predict([[random_values['HP'],random_values['こうげき'], random_values['ぼうぎょ'], random_values['とくこう'],random_values['とくぼう'],random_values['すばやさ']]])[0])
    random_values.update(Pokemon=str(closest_pokemon['名前']),pokemon_image=str(closest_pokemon['画像URL']),pokemon_height=float(closest_pokemon['高さ']),pokemon_weight=float(closest_pokemon['重さ']))

    return(random_values)
    return({'prediction':int(model.predict([[features.HP, features.こうげき, features.ぼうぎょ, features.とくこう,features.とくぼう,features.すばやさ]])[0]),'Pokemon':str(closest_pokemon['名前']),'pokemon_image':str(closest_pokemon['画像URL']),'HP':features.HP,'こうげき': features.こうげき, 'ぼうぎょ':features.ぼうぎょ, 'とくこう':features.とくこう,'とくぼう':features.とくぼう,'すばやさ':features.すばやさ})

    #1回featuresに格納して、それを返そうとしたけど失敗
    #features.update(prediction=str(model.predict([[features.HP, features.こうげき, features.ぼうぎょ, features.とくこう,features.とくぼう,features.すばやさ]])[0]),Pokemon=str(closest_pokemon['名前']),pokemon_image=str(closest_pokemon['画像URL']))
    #return(features)
    

