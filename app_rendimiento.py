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
    'Nombre': ['Andrés', 'Victor', 'Santiago', 'Anderson', 'Messi', 'Cristiano', 'Neymar'],
    'Nota_Lineal': [3.5, 4.0, 2.8, 4.5, 3.2, 2.0, 4.1],
    'Nota_Calculo': [3.0, 4.2, 3.1, 4.6, 3.0, 1.9, 4.3],
    'Asistencia': [90, 95, 75, 98, 85, 60, 92],
    'Horas_Estudio': [2, 3, 1, 4, 2, 0.5, 3.5],
    'Rendimiento': ['Medio', 'Alto', 'Bajo', 'Alto', 'Medio', 'Bajo', 'Alto']
})

# Mostrar datos
st.subheader("Conjunto de datos completo")
st.dataframe(data)

# Mostrar estadísticas descriptivas
st.subheader("Estadísticas descriptivas")
st.dataframe(data.describe())

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
# 3. Análisis de Matriz de Covarianza
# -----------------------------------------
st.header("3. Matriz de Covarianza")

# Calcular matriz de covarianza
cov_matrix = np.cov(X_scaled.T)
cov_df = pd.DataFrame(cov_matrix, columns=X.columns, index=X.columns)

st.subheader("Matriz de Covarianza")
st.dataframe(cov_df)

# Mapa de calor de covarianzas
st.subheader("Mapa de Calor de Covarianzas")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cov_df, annot=True, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)

# Explicación de la matriz de covarianza
with st.expander("¿Qué significa la matriz de covarianza?"):
    st.write("""
    - **Diagonal principal**: Contiene las varianzas de cada variable (valores alrededor de 1 porque los datos están estandarizados)
    - **Otras celdas**: Muestran cómo covarían las variables entre sí:
      - Valores positivos: Las variables aumentan/disminuyen juntas
      - Valores negativos: Cuando una aumenta, la otra disminuye
      - Cercano a cero: Poca relación lineal
    """)

# -----------------------------------------
# 4. Aplicar PCA para reducir la dimensionalidad
# -----------------------------------------
st.header("4. Análisis de Componentes Principales (PCA)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.subheader("Varianza explicada por cada componente principal")
var_exp = pca.explained_variance_ratio_
st.write(f"PC1: {var_exp[0]:.2%}")
st.write(f"PC2: {var_exp[1]:.2%}")
st.write(f"Total explicado: {(var_exp[0] + var_exp[1]):.2%}")

# Gráfico de varianza explicada
st.subheader("Varianza Explicada Acumulada")
fig, ax = plt.subplots()
ax.bar(range(1, len(var_exp)+1), var_exp, alpha=0.5, align='center', 
       label='Varianza individual explicada')
ax.step(range(1, len(var_exp)+1), np.cumsum(var_exp), where='mid',
        label='Varianza explicada acumulada')
ax.set_ylabel('Ratio de Varianza Explicada')
ax.set_xlabel('Componentes Principales')
ax.legend(loc='best')
st.pyplot(fig)

# -------------------------------
# 5. Clasificación con KNN
# -------------------------------
st.header("5. Modelo de Clasificación (KNN)")

# Widget para seleccionar parámetros
col1, col2 = st.columns(2)
with col1:
    n_vecinos = st.slider("Número de vecinos (K):", 1, 7, 3)
with col2:
    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3)

X_train, X_test, y_train, y_test, nombres_train, nombres_test = train_test_split(
    X_pca, y, nombres, test_size=test_size, random_state=42)

knn = KNeighborsClassifier(n_neighbors=n_vecinos)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Resultados
st.subheader("Resultados del Modelo")

# Obtener todas las clases posibles ordenadas
classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

# Mostrar métricas en columnas
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Matriz de Confusión:**")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, 
                        index=classes,
                        columns=classes)
    st.dataframe(cm_df.style.background_gradient(cmap='Blues'))

with col2:
    st.markdown("**Reporte de Clasificación:**")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

# Mostrar predicciones con nombre
st.subheader("Predicciones por estudiante")
predicciones = pd.DataFrame({
    'Nombre': nombres_test,
    'Rendimiento Real': y_test,
    'Rendimiento Predicho': y_pred,
    'Correcto': y_test == y_pred
})
st.dataframe(predicciones.style.apply(
    lambda x: ['background: lightgreen' if x else 'background: salmon' 
               for x in predicciones['Correcto']], 
    axis=0
))
# -----------------------------------------
# 6. Visualización de los resultados
# -----------------------------------------
st.header("6. Visualización de Resultados")

# Crear DataFrame para graficar
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Rendimiento'] = y.values
pca_df['Nombre'] = nombres.values

# Gráfico PCA
st.subheader("Distribución en Componentes Principales")
fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Rendimiento', s=100, ax=ax)

# Agregar nombres en el gráfico
for i in range(len(pca_df)):
    ax.text(pca_df['PC1'][i] + 0.02, pca_df['PC2'][i], pca_df['Nombre'][i], fontsize=9)

ax.set_title('Distribución de estudiantes según componentes principales')
ax.set_xlabel(f'Componente Principal 1 ({var_exp[0]:.2%})')
ax.set_ylabel(f'Componente Principal 2 ({var_exp[1]:.2%})')
ax.grid(True)
st.pyplot(fig)

# Gráfico de fronteras de decisión
st.subheader("Fronteras de Decisión del KNN")
fig, ax = plt.subplots(figsize=(10, 7))

# Crear meshgrid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Codificar etiquetas para el modelo
label_encoder = {label: idx for idx, label in enumerate(np.unique(y))}
y_encoded = np.array([label_encoder[label] for label in y])

# Entrenar modelo con etiquetas codificadas
knn_encoded = KNeighborsClassifier(n_neighbors=n_vecinos)
knn_encoded.fit(X_train, [label_encoder[label] for label in y_train])

# Predecir para cada punto del meshgrid
Z = knn_encoded.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotear contornos
contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')

# Scatter plot con etiquetas originales
unique_labels = np.unique(y)
colors = sns.color_palette(n_colors=len(unique_labels))
for i, label in enumerate(unique_labels):
    mask = (y == label)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               color=colors[i], s=100, 
               label=label, edgecolor='k')

ax.set_title(f'Fronteras de decisión (K={n_vecinos})')
ax.set_xlabel(f'Componente Principal 1 ({var_exp[0]:.2%})')
ax.set_ylabel(f'Componente Principal 2 ({var_exp[1]:.2%})')
ax.legend(title='Rendimiento')

# Barra de color para las regiones
cbar = plt.colorbar(contour, ax=ax)
cbar.set_ticks(np.linspace(0, len(unique_labels)-1, len(unique_labels)))
cbar.set_ticklabels(unique_labels)

st.pyplot(fig)

# Información adicional
st.sidebar.header("Acerca de")
st.sidebar.info("""
**Aplicación de Análisis de Rendimiento Estudiantil**

Esta aplicación muestra:
- Análisis exploratorio de datos
- Matriz de covarianza
- Reducción de dimensionalidad con PCA
- Modelo de clasificación KNN

Desarrollado con:
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib/Seaborn
""")

st.sidebar.header("Parámetros del Modelo")
st.sidebar.write(f"**Vecinos KNN:** {n_vecinos}")
st.sidebar.write(f"**Tamaño prueba:** {test_size:.0%}")
st.sidebar.write(f"**Varianza explicada:** {(var_exp[0] + var_exp[1]):.2%}")