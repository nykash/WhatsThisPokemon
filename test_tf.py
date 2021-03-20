import cnn
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import tqdm
from tensorflow.keras import models, layers, activations, optimizers, losses

df = pd.read_csv("PokemonDataset/pokemon.csv")
df.columns = ["row", "gen", "game", "dir", "pokedex"]
df_Train = df.query('gen == "gen1" or gen == "gen3" or gen == "gen4"')
df_Test = df.query('gen == "gen5"')
df_Test = df_Test.query('pokedex < 151')
df_Train = df_Train.query('pokedex < 151')

x_train_l = []
y_train_l = []

for i in range(len(df_Test)):
    row = df_Train.iloc[i]
    img = cv2.imread("PokemonDataset/"+row[3], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))

    x_train_l.append(img.reshape((32, 32, 1)))
    y_new = np.zeros(151)
    y_new[row[4]-1] = 1
    y_train_l.append(y_new)

x_train = np.array(x_train_l)
y_train = np.array(y_train_l)

model = models.load_model("best_classifier_151.h5")
model.evaluate(x_train, y_train)

cv2.imshow("bruv", x_train[4])
print(model(x_train[4][np.newaxis, ...]))
print(y_train[4])