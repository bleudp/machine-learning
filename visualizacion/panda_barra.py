import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Ciudades_Europa = ["Madrid", "Valencia", "Paris", "Londres", "Roma"]
pos = np.arange(len(Ciudades_Europa))
Habitantes_millones = [8, 4, 16, 9]

Habitantes = list(zip(Ciudades_Europa, Habitantes_millones))

Europa_dataframe = pd.DataFrame(data = Habitantes, columns=["Ciudades", "Habitantes"])
Europa_dataframe.plot.bar()

plt.xticks(pos, Ciudades_Europa)
plt.show()
