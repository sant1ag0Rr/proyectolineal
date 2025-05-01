import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# 1. Cargar y explorar los datos
# ------------------------------

data = pd.DataFrame({
    'Nombre': ['Ana', 'Luis', 'Carlos', 'Sofía', 'María', 'Pedro', 'Elena'],
    'Nota_Matematicas': [3.5, 4.0, 2.8, 4.5, 3.2, 2.0, 4.1],
    'Nota_Lengua': [3.0, 4.2, 3.1, 4.6, 3.0, 1.9, 4.3],
    'Asistencia': [90, 95, 75, 98, 85, 60, 92],
    'Horas_Estudio': [2, 3, 1, 4, 2, 0.5, 3.5],
    'Rendimiento': ['Medio', 'Alto', 'Bajo', 'Alto', 'Medio', 'Bajo', 'Alto']
})

print("Vista general del conjunto de datos:")
print(data.head())

# -------------------------------
# 2. Preprocesamiento de los datos
# -------------------------------

# Separar características (X) y etiquetas (y)
X = data.drop(['Nombre', 'Rendimiento'], axis=1)
y = data['Rendimiento']
nombres = data['Nombre']

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 3. Aplicar PCA para reducir la dimensionalidad
# -----------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nVarianza explicada por cada componente principal:")
print(np.round(pca.explained_variance_ratio_, 4))

# -------------------------------
# 4. Clasificación con KNN
# -------------------------------

X_train, X_test, y_train, y_test, nombres_train, nombres_test = train_test_split(
    X_pca, y, nombres, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Resultados
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))

# Mostrar predicciones con nombre
print("\nPredicciones por estudiante:")
for nombre, real, pred in zip(nombres_test, y_test, y_pred):
    print(f"{nombre}: Real = {real}, Predicho = {pred}")

# -----------------------------------------
# 5. Visualización de los resultados
# -----------------------------------------

# Crear DataFrame para graficar
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Rendimiento'] = y.values
pca_df['Nombre'] = nombres.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Rendimiento', s=100)

# Agregar nombres en el gráfico
for i in range(len(pca_df)):
    plt.text(pca_df['PC1'][i] + 0.02, pca_df['PC2'][i], pca_df['Nombre'][i], fontsize=9)

plt.title('Distribución de estudiantes según componentes principales')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()
