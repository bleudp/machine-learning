import pandas as pd
import matplotlib.pyplot as plt

# Lectura del conjunto de datos
df = pd.read_csv("dataframe.csv", names=["CPU", "IO", "Class"])

# Separamos el conjunto de datos en caracteristicas legitimas y anomalas
df_legit = df[df["Class"] == 0]
df_anom = df[df["Class"] == 1]

# Separamos el conjunto de datos en train set y test set
legit_len = len(df_legit)
anom_len = len(df_anom)

df_anom_train = df_anom[: (anom_len // 2)]
df_anom_test = df_anom[(anom_len // 2) + 1: anom_len]

train_limit = (legit_len * 60) // 100

df_legit_train = df_legit[:train_limit]
df_legit_test = df_legit[train_limit +1: legit_len]

# Concatenamos los sets de train y test para utilizarlos para detectar las
# anomalias

df_train = pd.concat([df_legit_train, df_anom_train], axis = 0)
df_test = pd.concat([df_legit_test, df_anom_test], axis = 0)

# Guardamos la clase de nuestros ejemplos
class_train = df_train["Class"]
class_test = df_test["Class"]

df_train.drop(labels=["Class"], axis = 1, inplace = True)
df_test.drop(labels=["Class"], axis = 1, inplace = True)

# Ejecucion del algoritmo
isolation = IsolationForest(max_features = 2, contamination = 0.132)
isolation.fit(df_train)
predictions = isolation.predict(df_test)

# Representacion grafica de los resultados
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(df_test["CPU"], df_test["IO"], marker="o", color="lightgrey")

for i, global_i in enumerate(df_test["CPU"].index):
    if class_test.loc[global_i] == 1:
        ax.annotate('o', (df_test["CPU"].loc[global_i], df_test["IO"].loc[global_i]), fontsize = 20, color = "black")
    if predictions[i] == -1:
        ax.annotate("*", (df_test["CPU"].loc[global_i], df_test["IO"].loc[global_i]), fontsize = 20, color = "black")
plt.ylabel("DISK IO")
plt.xlabel("% CPU")
plt.show()
