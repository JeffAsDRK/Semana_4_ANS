import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

# Cargar los datasets
data_train = np.loadtxt("ECG5000/ECG5000_TRAIN", delimiter=",")
X_train = to_time_series_dataset(data_train[:, 1:])
y_train = data_train[:, 0].astype(int)

data_test = np.loadtxt("ECG5000/ECG5000_TEST", delimiter=",")
X_test = to_time_series_dataset(data_test[:, 1:])
y_test = data_test[:, 0].astype(int)

# Escalar las series temporales para la media y la varianza
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Estandarizar para tener media 0 y varianza 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Estadísticas básicas
print("Número de series temporales:", len(data_train))
print("Número de clases únicas:", len(np.unique(data_train[:,0])))
print("Longitud de las series temporales:", len(data_train[0,1:]))

# Número de ejemplos por clase en el conjunto de entrenamiento
print("Número de series temporales en clase 1.0:", len(data_train[data_train[:,0]==1.0]))
print("Número de series temporales en clase 2.0:", len(data_train[data_train[:,0]==2.0]))

# Inicializar el modelo K-Shape
n_clusters = len(np.unique(y_train))
ks = KShape(n_clusters=n_clusters, n_init=1, random_state=0)

# Ajustar el modelo
ks.fit(X_train_scaled)

# Predecir los clusters
y_pred = ks.predict(X_train_scaled)

# Visualizar los centroides de los clusters y las series originales
for yi in range(n_clusters):
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Cluster {yi + 1} Series Originales")
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title(f"Cluster {yi + 1} Series Escaladas con Centroide")
    for xx in X_train_scaled[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.grid(True)
    
    plt.show()
