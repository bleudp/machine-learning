import pandas as pd
from matplotlib import pyplot

# Se importa el conjunto de datos
datos = pd.read_csv('/home/blueudp/Desktop/machine-learning/UDP_SCANNER/capture.csv')
print(datos.head())

# Representacion grafica del conjunto de datos

pyplot.rcParams['figure.figsize'] = (12, 8)
feature1 = datos['Length'].values
feature2 = [10] * len(feature1)

pyplot.scatter(feature1, feature2, c='black', s=5)
pyplot.show()
