import pandas as pd
import numpy as np

Ciudades_Europa = ["Madrid", "Valencia", "Paris", "Londres", "Roma"]
Habitantes_millones = [8,4,12,16,9]

Habitantes = list(zip(Ciudades_Europa, Habitantes_millones))

Europa_Dataframe = pd.DataFrame(data = Habitantes, columns=["Ciudades", "Habitantes"])

print(Europa_Dataframe)
