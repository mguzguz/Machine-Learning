# Informe de clasificación de DATASET IRIS - Regresión Lineal 
*Autoras:* María Camila Guzmán Bello y Ana María Casallas Ariza
*Materia:* Machine Learning 802
*Fecha:* 2025-09-13

# Clasificación de flores Iris con Regresión Lineal

## Procedimiento
1. Carga del dataset  
   Se utiliza el conjunto de datos Iris de scikit-learn, que contiene 150 muestras de tres especies de flores: setosa, versicolor y virginica.  
   Cada muestra tiene 4 características:  
   - Largo del sépalo  
   - Ancho del sépalo  
   - Largo del pétalo  
   - Ancho del pétalo  

2. Preparación de los datos  
   - Se dividen los datos en entrenamiento (70%) y prueba (30%).  
   - Se normalizan las variables mediante StandardScaler para que todas tengan la misma escala.  

3. Entrenamiento del modelo  
   - Se aplica un esquema One-vs-Rest (OvR), entrenando un regresor lineal para cada clase.  
   - Cada modelo responde a la pregunta: ¿La flor pertenece a esta clase (1) o no (0)?  
   - El algoritmo usado para entrenar es LinearRegression de scikit-learn.  

4. Predicción de nuevas flores  
   - El usuario ingresa por consola los 4 valores de una flor.  
   - Cada regresor calcula un puntaje y se elige la clase con el valor más alto como predicción final.  

5. Evaluación y gráficos  
   - Se calcula la exactitud del modelo comparando las predicciones con los valores reales.  
   - Para cada regresor se genera una gráfica de valores reales vs. predichos, mostrando también:  
     - La ecuación de la recta entrenada.  
     - El Error Cuadrático Medio (ECM) como métrica de desempeño.  



## Algoritmo utilizado
- Regresión Lineal (Linear Regression) en enfoque One-vs-Rest (OvR).  
- Implementado únicamente con *scikit-learn* para el entrenamiento y evaluación.  


## Librerías empleadas
- scikit-learn: entrenamiento y evaluación del modelo de regresión lineal.  
- NumPy: operaciones matemáticas auxiliares.  
- Matplotlib: visualización de los resultados y gráficas del modelo.
