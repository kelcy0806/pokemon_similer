import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

st.title('pokemon similer')
st.write('このサイトでは、あなたのスタイル・性格からどんな体型のポケモンなのか？どんな戦い方していくのかを提示してくれます')

st.sidebar.header('Input Features')
your_height = st.sidebar.slider('your height (m)', min_value=1.00, max_value=2.00, step=0.01)
your_weight = st.sidebar.slider('your weight (kg)', min_value=0.0, max_value=200.0, step=0.1)

st.sidebar.write('あなたの性格')
hp = st.sidebar.slider('情熱的< - >静観的', min_value=-1.00, max_value=1.00, step=0.2)
attack = st.sidebar.slider('大胆< - >慎重', min_value=-1.00, max_value=1.00, step=0.2)
bold = st.sidebar.slider('せっかち< - >おっとり', min_value=-1.00, max_value=1.00, step=0.2)
clitical = st.sidebar.slider('C', min_value=-1.00, max_value=1.00, step=0.2)
deffence = st.sidebar.slider('D', min_value=-1.00, max_value=1.00, step=0.2)
spped = st.sidebar.slider('S', min_value=-1.00, max_value=1.00, step=0.2)

your_feature ={
    "your_height": your_height,
    "your_weight": your_weight,
    "HP":hp,
    "こうげき":attack,
    "ぼうぎょ":bold,
    "とくこう":clitical,
    "とくぼう":deffence,
    "すばやさ":spped,
}

targets = ['先陣きって荒らしまくる切り込み隊長','味方を引っ張るリーダー','後続を支援する起点作りの職人','じわじわと敵を追い詰める仕事人','なんでもこなせるマルチプレイヤー','単体でつよい絶対的エース','敵の攻撃を受けて、サイクルを回すタンク','必殺のコンボで相手を倒す暗殺者','何してくるかわからないミステリアスな策略家']


if st.sidebar.button("GEUSS!!"):
    # 入力された説明変数の表示
    st.write('## Input Value')
    your_df = pd.DataFrame(your_feature, index=["あなた"])
    st.write(your_df[['your_height','your_weight']])

    # 予測の実行
    #response = requests.post("https://pokemon-predict.onrender.com/guess", json=your_feature) #デプロイ用
    response = requests.post("http://localhost:5001/guess", json=your_feature) #ローカル用
    battle_style = response.json()["prediction"]
    all = response.json()

    pokemon = response.json()["Pokemon"]
    pokemon_image = response.json()["pokemon_image"]

    # 予測結果の出力
    st.write('# 身長・体重から推測')
    st.write('あなたに似ているポケモンはきっと',str(pokemon),'です!')
    
    # ポケモンの身長・体重を出力
    pokemon_df = pd.DataFrame(all, index=[pokemon])
    pokemon_df = pokemon_df[['pokemon_height','pokemon_weight']]
    st.write(pokemon_df)

    # 予測画像の出力
    # URLから画像を取得
    response = requests.get(str(pokemon_image))
    # 取得した画像データをPIL Imageに変換
    img = Image.open(BytesIO(response.content))

    # 画像を表示
    plt.imshow(img)
    plt.axis('off')  # 軸を非表示にする
    st.pyplot(plt)

    # 予測結果の出力(種族値)
    st.write('# あなたの種族値')
    potential_data = pd.DataFrame(all, index=["あなた"])
    potential_data_sum = potential_data.iloc[:,:7]
    potential_data = potential_data.iloc[:,:6]
    st.write(potential_data_sum)


    #レーダーチャート図の記載
    #def radar_chart():
    values =list(potential_data.iloc[0])
    angles = np.linspace(0, 2 * np.pi, len(values) + 1)
    #labels = list(potential_data.columns)
    labels = ['HP','attack','deffence','special attack','special deffence','speed']

    values.append(values[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, polar = True)

    ax.plot(angles, values)

    ax.fill(angles, values, alpha=0.2)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=15)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.tick_params(labelsize=15)

    ax.set_title('Status', fontsize=20)

        # レーダーチャートを表示
    st.write('レーダーチャート')
    st.pyplot(plt)

    # 予測結果の出力(戦闘スタイル)
    st.write('# 戦闘スタイル')
    st.write('あなたの戦闘スタイルはきっと',str(targets[int(battle_style)]),'です')
    
    # 以下いらないコードなのでテキスト化

    # 予測結果の表示
    #st.write('## Prediction')
    #st.write(pokemon)

    #st.write(str(pokemon_image))
    #st.write(repr(str(pokemon_image)))
    #image = Image.open(repr(str(pokemon_image)))
    #image = Image.open("./1.png")
    #image = Image.open("https://github.com/PokeAPI/sprites/blob/master/sprites/pokemon/1.png")
    #st.image(image, caption='サンプル',use_column_width=True)