import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import matplotlib.pyplot as plt

# Generar datos sintéticos
X, y = datasets.make_moons(n_samples=300, noise=0.2)

# Generar datos sintéticos complejos
#X, y = datasets.make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)

# Simulamos que algunos datos no están etiquetados
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y)) < 0.7  # Etiquetamos al azar el 70% de los datos
labels = np.copy(y)
labels[random_unlabeled_points] = -1  # -1 indica dato no etiquetado

# Crear y entrenar el modelo de label propagation
prop_model = LabelPropagation()
prop_model.fit(X, labels)

# Crear y entrenar el modelo de label spreading
spread_model = LabelSpreading()
spread_model.fit(X, labels)

# Predecir las etiquetas para todo el conjunto de datos
output_labels_prop = prop_model.transduction_
output_labels_spread = spread_model.transduction_

# Visualización de resultados
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(X[labels != -1, 0], X[labels != -1, 1], c=y[labels != -1], marker='o', label='Datos etiquetados', edgecolors='k')
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c=output_labels_prop[labels == -1], marker='s', label='Datos no etiquetados', alpha=0.5, edgecolors='k')
plt.title("Label Propagation")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X[labels != -1, 0], X[labels != -1, 1], c=y[labels != -1], marker='o', label='Datos etiquetados', edgecolors='k')
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c=output_labels_spread[labels == -1], marker='s', label='Datos no etiquetados', alpha=0.5, edgecolors='k')
plt.title("Label Spreading")
plt.legend()

plt.show()
