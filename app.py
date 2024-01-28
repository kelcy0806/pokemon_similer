import streamlit as st
import pandas as pd
import requests
from PIL import Image

st.title('pokemon similer')

st.sidebar.header('Input Features')
your_height = st.sidebar.slider('your height (m)', min_value=1.00, max_value=2.00, step=0.01)
your_weight = st.sidebar.slider('your weight (kg)', min_value=0.0, max_value=200.0, step=0.1)

your_feature ={
    "your_height": your_height,
    "your_weight": your_weight,
}



if st.sidebar.button("GEUSS!!"):
    # 入力された説明変数の表示
    st.write('## Input Value')
    your_df = pd.DataFrame(your_feature, index=["data"])
    st.write(your_df)

    # 予測の実行
    response = requests.post("https://pokemon-predict.onrender.com/guess", json=your_feature) #デプロイ用
    #response = requests.post("http://localhost:8000/guess", json=your_feature) #ローカル用
    pokemon = response.json()["Pokemon"]
    pokemon_image = response.json()["pokemon_image"]


    # 予測結果の出力
    st.write('## Result')
    st.write('あなたに似ているポケモンはきっと',str(pokemon),'です!')

    # 予測結果の表示
    st.write('## Prediction')
    st.write(pokemon)

    # 予測画像の出力
    st.write(str(pokemon_image))
    st.write(repr(str(pokemon_image)))
    #image = Image.open(repr(str(pokemon_image)))
    image = Image.open("./1.png")
    #image = Image.open("https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/1.png")
    st.image(image, caption='サンプル',use_column_width=True)


