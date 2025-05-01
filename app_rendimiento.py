import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Configuración de la página
st.set_page_config(page_title="Análisis de Rendimiento Estudiantil", layout="wide")

# Título de la aplicación
st.title("Análisis de Rendimiento Estudiantil")

# ------------------------------
# 1. Cargar y explorar los datos
# ------------------------------
st.header("1. Datos de Estudiantes")

data = pd.DataFrame({
    'Nombre': ['Ana', 'Luis', 'Carlos', 'Sofía', 'María', 'Pedro', 'Elena'],
    'Nota_Matematicas': [3.5, 4.0, 2.8, 4.5, 3.2, 2.0, 4.1],
    'Nota_Lengua': [3.0, 4.2, 3.1, 4.6, 3.0, 1.9, 4.3],
    'Asistencia': [90, 95, 75, 98, 85, 60, 92],
    'Horas_Estudio': [2, 3, 1, 4, 2, 0.5, 3.5],
    'Rendimiento': ['Medio', 'Alto', 'Bajo', 'Alto', 'Medio', 'Bajo', 'Alto']
})

# Mostrar datos
st.subheader("Conjunto de datos completo")
st.dataframe(data)

# -------------------------------
# 2. Preprocesamiento de los datos
# -------------------------------
st.header("2. Preprocesamiento de Datos")

# Separar características (X) y etiquetas (y)
X = data.drop(['Nombre', 'Rendimiento'], axis=1)
y = data['Rendimiento']
nombres = data['Nombre']

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("Datos estandarizados")
st.dataframe(pd.DataFrame(X_scaled, columns=X.columns))

# -----------------------------------------
# 3. Aplicar PCA para reducir la dimensionalidad
# -----------------------------------------
st.header("3. Análisis de Componentes Principales (PCA)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.subheader("Varianza explicada por cada componente principal")
st.write(np.round(pca.explained_variance_ratio_, 4))

# -------------------------------
# 4. Clasificación con KNN
# -------------------------------
st.header("4. Modelo de Clasificación (KNN)")

# Seleccionar número de vecinos
n_vecinos = st.slider("Seleccione el número de vecinos para KNN:", 1, 7, 3)

X_train, X_test, y_train, y_test, nombres_train, nombres_test = train_test_split(
    X_pca, y, nombres, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=n_vecinos)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Resultados
st.subheader("Resultados del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Matriz de Confusión:**")
    st.write(confusion_matrix(y_test, y_pred))

with col2:
    st.markdown("**Reporte de Clasificación:**")
    st.write(classification_report(y_test, y_pred, zero_division=0))

# Mostrar predicciones con nombre
st.subheader("Predicciones por estudiante")
predicciones = pd.DataFrame({
    'Nombre': nombres_test,
    'Rendimiento Real': y_test,
    'Rendimiento Predicho': y_pred
})
st.dataframe(predicciones)

# -----------------------------------------
# 5. Visualización de los resultados
# -----------------------------------------
st.header("5. Visualización de Resultados")

# Crear DataFrame para graficar
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Rendimiento'] = y.values
pca_df['Nombre'] = nombres.values

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Rendimiento', s=100, ax=ax)

# Agregar nombres en el gráfico
for i in range(len(pca_df)):
    ax.text(pca_df['PC1'][i] + 0.02, pca_df['PC2'][i], pca_df['Nombre'][i], fontsize=9)

ax.set_title('Distribución de estudiantes según componentes principales')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.grid(True)

st.pyplot(fig)

# Información adicional
st.sidebar.header("Acerca de")
st.sidebar.info("""
Esta aplicación web muestra un análisis de rendimiento estudiantil utilizando:
- PCA para reducción de dimensionalidad
- KNN para clasificación
""")