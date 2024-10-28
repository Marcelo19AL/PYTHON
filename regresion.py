import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos
data = pd.read_csv('csv/supermarket_sales - Sheet1.csv')

# Seleccionar las columnas relevantes para el análisis
X = data[['Unit price', 'Quantity', 'Tax 5%', 'Rating']]
y = data['Total']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar los coeficientes, el error cuadrático medio y el puntaje R^2
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)



import matplotlib.pyplot as plt

# Generar el gráfico de residuos
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(y_pred, y_test - y_pred)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')

# Gráfico de valores predichos vs valores reales
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Predichos vs Reales')

# Histograma de residuos
plt.subplot(1, 3, 3)
plt.hist(y_test - y_pred, bins=20, edgecolor='k')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')

# Mostrar todos los gráficos juntos
plt.tight_layout()
plt.show()
