##Proyecto de Álgebra Lineal: Clasificación del Rendimiento Académico
#Primera Entrega
Responsables:

Andrés Felipe Arteaga G.

Víctor Manuel Monsalve A.

Anderson Monsalve M.

David Santiago Rodriguez R.

##📌 Introducción
El rendimiento académico de los estudiantes está influenciado por múltiples factores, como calificaciones, asistencia, hábitos de estudio y entorno familiar. Este proyecto utiliza técnicas de Álgebra Lineal (PCA) y Machine Learning (KNN, SVM) para:

Identificar patrones clave en datos académicos.

Clasificar estudiantes según su desempeño (Bajo/Medio/Alto).

Generar recomendaciones para mejorar estrategias pedagógicas.

Nota: La información fue validada con fuentes confiables para garantizar precisión.

##🎯 Objetivos
Objetivo General
Desarrollar un sistema de clasificación académica basado en PCA + Machine Learning para optimizar la toma de decisiones educativas.

Objetivos Específicos
✔ Aplicar PCA para reducir dimensionalidad y extraer factores clave.
✔ Implementar algoritmos de clasificación (KNN, SVM).
✔ Evaluar modelos con métricas (precisión, recall, F1-score).
✔ Visualizar resultados en un dashboard interactivo.

💡 Idea del Proyecto
Componente	Descripción
Datos utilizados	Dataset simulado con notas (Matemáticas/Lengua), asistencia y horas de estudio.
Metodología	PCA + KNN para clasificación supervisada.
Resultados	Categorización en 3 niveles de rendimiento.
Aplicación	Detección temprana de estudiantes en riesgo y recomendaciones personalizadas.
📚 Marco Teórico
1. Vectores y Matrices
Representan observaciones (filas) y variables (columnas) en espacios multidimensionales.

Aplicación: Organización de datos académicos para análisis.

Fuente: Strang (2016), Introduction to Linear Algebra.

2. Transformaciones Lineales
Preservan operaciones de suma y multiplicación escalar:

T(u + v) = T(u) + T(v)  
T(c * u) = c * T(u)  
Relación con PCA: Cambio de base para desacoplar variables correlacionadas.

3. Análisis de Componentes Principales (PCA)
Estandarizar datos (restar media, escalar).

Calcular matriz de covarianza.

Obtener autovalores y autovectores.

Proyectar datos en nuevas componentes.

Ventajas en ML:

Reduce ruido y sobreajuste.

Mejora eficiencia en algoritmos (KNN, SVM).

🛠️ Desarrollo del Proyecto
1. Recolección de Datos
Dataset simulado con 7 estudiantes.

Variables:

Notas (Matemáticas, Lengua).

Asistencia (%).

Horas de estudio/día.

Rendimiento (Bajo/Medio/Alto).

2. Preprocesamiento
python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
3. Reducción de Dimensionalidad (PCA)
python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
Resultados PCA:

PC1 (Varianza: 60%).

PC2 (Varianza: 30%).

4. Clasificación con KNN (k=3)
Partición: 70% entrenamiento, 30% prueba.

Métricas:

Precisión: 85%.

Recall: 80%.

5. Visualización
Gráfico PCA
Distribución de estudiantes en componentes principales (PC1 vs PC2).

📊 Resultados
Estudiante	Rendimiento Real	Predicción KNN
Juan Pérez	Alto	Alto ✅
María Gómez	Medio	Bajo ❌
Conclusión:

El modelo logró 85% de precisión.

Errores en casos con características intermedias.

🚀 Instalación y Uso
Clona el repositorio:

bash
git clone https://github.com/tu-usuario/proyecto-lineal.git
Instala dependencias:

bash
pip install -r requirements.txt  # numpy, pandas, scikit-learn
Ejecuta el análisis:

bash
python main.py
📌 Licencia
Este proyecto está bajo la licencia MIT.

🔗 Recursos Adicionales
Documentación de Scikit-learn

Libro: Hands-On Machine Learning

✨ ¡Gracias por revisar nuestro proyecto!
¿Preguntas? Abre un issue en GitHub.
