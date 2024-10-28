import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
data = pd.read_csv('csv/CarPrice_Assignment.csv')  # Cambia esta ruta al archivo correspondiente
print(data.head())  # Para ver las primeras filas del dataset

# Preprocesamiento de datos
# Seleccionamos solo las columnas de interés
data = data[['enginesize', 'horsepower', 'price']]

# Separar variables independientes y dependiente
X = data[['enginesize', 'horsepower']]  # Variables predictoras
y = data['price']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)
print("Coeficientes del modelo:", model.coef_)
print("Intercepto:", model.intercept_)

# Gráficos interpretativos

# 1. Importancia de las Variables
plt.figure(figsize=(8, 6))
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
sns.barplot(x=coef_df['Coeficiente'], y=coef_df.index, palette='viridis')
plt.title('Importancia de las Variables en el Precio')
plt.xlabel('Valor del Coeficiente')
plt.ylabel('Variable')
plt.show()

# 2. Valores Predichos vs Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Predichos vs Valores Reales')
plt.show()

# 3. Distribución de los Residuos
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='k')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.show()

