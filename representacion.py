import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['largo sepalo', 'ancho sepalo', 'largo petalo', 'ancho petalo', 'especie'])
features = ['largo sepalo', 'ancho sepalo', 'largo petalo', 'ancho petalo']
print("dataset")
print(df.head())
print("dataset normalizado")
x = df.loc[:,features].values
y = df.loc[:,['especie']].values
x = StandardScaler().fit_transform(x)
print(pd.DataFrame(data = x, columns = features).head())
print("dataset reducido 2 D")
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame( data = principalComponents
                                , columns = ['Componente principal 1', 'Componente principal 2'])
print(principalDf.head(5))


df[['especie']].head()
finalDf = pd.concat([principalDf, df[['especie']]], axis = 1)
finalDf.head(5)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente Principal 1', fontsize = 15)
ax.set_ylabel('Componente Principal 2', fontsize = 15)
ax.set_title('PCA 2 Componentes', fontsize=20)
especies = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['black', 'gray', 'silver']



for especie, color in zip(especies,colors):
	indicesToKeep = finalDf['especie'] == especie
	ax.scatter(finalDf.loc[indicesToKeep, 'Componente principal 1'], finalDf.loc[indicesToKeep, 'Componente principal 2']
	, c = color
	, s = 50 )
ax.legend(especies)
ax.grid()
