import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import models, layers, activations, optimizers, losses

df = pd.read_csv("PokemonDataset/pokemon.csv")
df.columns = ["row", "gen", "game", "dir", "pokedex"]
df_Train = df.query('gen == "gen1" or gen == "gen2" or gen == "gen3" or gen == "gen4"')
df_Test = df.query('gen == "gen2"')
df_Test = df_Test.query('pokedex <= 493')
df_Train = df_Train.query('pokedex <= 493')

x_train_l = []
y_train_l = []

for i in range(len(df_Train)):
    row = df_Train.iloc[i]
    img = cv2.imread("PokemonDataset/"+row[3], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))

    x_train_l.append(img.reshape((32, 32, 1)))
    y_new = np.zeros(493)
    y_new[row[4]-1] = 1
    y_train_l.append(y_new)

x_train = np.array(x_train_l)
y_train = np.array(y_train_l)

model = models.Sequential([
    layers.Conv2D(256, (3, 3), input_shape=(32, 32, 1), padding="same", activation="relu"),
    layers.MaxPool2D((2, 2), padding="same"),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.MaxPool2D((2, 2), padding="same"),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPool2D((2, 2), padding="same"),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.Flatten(),
    layers.Dense(200),
    layers.Dense(493, activation="softmax")
])

model.compile(loss=losses.CategoricalCrossentropy(), metrics=["acc"])

history = model.fit(np.divide(x_train, 255), y_train, epochs=50)
model.save("best_classifier_493.h5", save_format="h5")