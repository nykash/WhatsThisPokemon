import pandas as pd
import cv2
import os

def get_pokedex_number(name):
    temp = ""
    for i in range(len(name)):
        if(name[i].isdigit()):
            temp += name[i]

    return int(temp)


df = pd.DataFrame()

pd.set_option("display.max_rows", None, "display.max_columns", None)

d = {
    "generation": [],
    "game": [],
    "dir": [],
    "pokedex number": []
}

for gen in os.listdir("PokemonDataset"):
    for game in os.listdir("PokemonDataset/"+gen):
        mons = os.listdir("PokemonDataset/"+gen+"/"+game)
        for mon in mons:

            d["generation"].append(gen)
            d["game"].append(game)
            d["dir"].append(gen+"/"+game+"/"+mon)
            d["pokedex number"].append(get_pokedex_number(mon))

df["generation"] = d["generation"]
df["game"] = d["game"]
df["dir"] = d["dir"]
df["pokedex number"] = d["pokedex number"]

df.to_csv("PokemonDataset/pokemon.csv")
print(df)