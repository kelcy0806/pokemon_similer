
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# データ準備
df = pd.read_csv("pokemon_practice.csv")
df = df.dropna(subset=['戦術'])
potential_data =df.iloc[:,9:15]
target_data =df['戦術']


# モデル構築
model = RandomForestClassifier()
model.fit(potential_data, target_data)

# モデルの保存
import pickle
pickle.dump(model, open('models/pokemon_model', 'wb'))