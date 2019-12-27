from copy import deepcopy
import pandas as pd
from matplotlib import pyplot
import numpy as np

# distancia euclidea

def distancia(a, b, ax=1):
	return np.linalg.norm(a - b, axis = ax)

# Se importa el conjunto de datos

datos = pd.read_csv('/home/blueudp/Desktop/machine-learning/UDP_SCANNER/capture.csv')
print(datos.head())

# Representacion grafica del conjunto de datos

pyplot.rcParams['figure.figsize']  = (12, 8)

feature1 = datos['Length'].values
feature2 = [10] * len(feature1)

pyplot.scatter(feature1, feature2, c='black', s=5)
pyplot.show()

# Iniciacion de los centroides
# Numero de clusters centroides
K = 4

# Inicializacion de coordenadas Y y X para cada cluster centroid
C_x = np.random.randint(0, np.max(feature1), size=K)
C_y = [10] * K

# Representacion grafica

pyplot.scatter(feature1, feature2, c='black', s=5)
pyplot.scatter(C_x, C_y, marker='*', c='b', s=600)
pyplot.show()

# Inicio k-means
# Agrupacion de los datos en matrices

X = np.array(list(zip(feature1, feature2)))
C = np.array(list(zip(C_x, C_y)), dtype = np.float32)

# Variable para actualizar el valor de los centroides cuando se actualizen

C_anterior = np.zeros(C.shape)

# Etiquetas de los clusters
clusters = np.zeros(len(X))

# Se calcula la distancia entre los nuevos centroids y los anteriores
dist = distancia(C, C_anterior, None)

while dist != 0:
	# Se asigna cada valor al cluster mas cercano
	for i in range(len(X)):
		distancias = distancia(X[i], C)
		# Se elige la menor
		c_min = np.argmin(distancias)
		clusters[i] = c_min

		# Asignacion de los nuevos valores a los centroids
		for i in range(K):
			datos_asignados = [X[j] for j in range(len(X)) if clusters[j] == i]
			# Se calcula la media de los elementos asignados
			C[i] = np.mean(datos_asignados, axis = 0)
			# Se guardan los valores anteriores
			C_anterior = deepcopy(C)
			# Se comprueba si la posicion de los centroids ha variado
			dist = distancia(C, C_anterior, None)
# Representacion grafica del resultado
COLORES = ['y', 'r', 'g', 'b']
fig, ax = pyplot.subplots()
for i in range(K):
	datos_asignados = np.array([X[j] for j in range(len(X)) if clusters[j] ==  i])
	ax.scatter(datos_asignados[:,0], datos_asignados[:,1], s=5, c=COLORES[i])
ax.scatter(C[:,0], C[:,1], marker='*', s=600, c='b')
pyplot.show()
