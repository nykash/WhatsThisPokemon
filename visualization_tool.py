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


for i in range(len(df_Test)):
    row = df_Train.iloc[i]
    img = cv2.imread("PokemonDataset/"+row[3], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))

    cv2.imshow("image", img)
    cv2.waitKey(1000)
