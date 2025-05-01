# Proyecto de Álgebra Lineal: Clasificación del Rendimiento Académico  

## Primera Entrega  

### Responsables:  
- Andrés Felipe Arteaga G.  
- Víctor Manuel Monsalve A.  
- Anderson Monsalve M.  
- David Santiago Rodriguez R.  

---

## 📌 Introducción  
El rendimiento académico de los estudiantes está influenciado por múltiples factores, como **calificaciones, asistencia, hábitos de estudio y entorno familiar**. Este proyecto utiliza técnicas de **Álgebra Lineal (PCA)** y **Machine Learning (KNN, SVM)** para:  
- Identificar patrones clave en datos académicos  
- Clasificar estudiantes según su desempeño (**Bajo/Medio/Alto**)  
- Generar recomendaciones para mejorar estrategias pedagógicas  

**Nota:** La información fue validada con fuentes confiables para garantizar precisión.  

---

## 🎯 Objetivos  

### Objetivo General  
Desarrollar un sistema de clasificación académica basado en **PCA + Machine Learning** para optimizar la toma de decisiones educativas.  

### Objetivos Específicos  
- ✔ Aplicar PCA para reducir dimensionalidad y extraer factores clave  
- ✔ Implementar algoritmos de clasificación (**KNN, SVM**)  
- ✔ Evaluar modelos con métricas (**precisión, recall, F1-score**)  
- ✔ Visualizar resultados en un **dashboard interactivo**  

---

## 💡 Idea del Proyecto  

| Componente       | Descripción                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Datos**        | Dataset simulado con notas (Matemáticas/Lengua), asistencia y horas de estudio |
| **Metodología**  | PCA + KNN para clasificación supervisada                                    |
| **Resultados**   | Categorización en 3 niveles de rendimiento                                 |
| **Aplicación**   | Detección temprana de estudiantes en riesgo y recomendaciones personalizadas |

---

## 📚 Marco Teórico  

### 1. Vectores y Matrices  
- Representan **observaciones (filas)** y **variables (columnas)** en espacios multidimensionales  
- **Aplicación:** Organización de datos académicos para análisis  
- **Fuente:** Strang (2016), *Introduction to Linear Algebra*  

### 2. Transformaciones Lineales  
```math
T(u + v) = T(u) + T(v)  
T(c \cdot u) = c \cdot T(u)
