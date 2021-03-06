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
