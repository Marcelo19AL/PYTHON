import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el archivo CSV
data = pd.read_csv('csv/Data.csv')

# Asegúrate de que el archivo tiene las columnas correctas para el análisis
print(data.head())

# Definir las variables dependiente e independientes
# Suponiendo que las columnas son 'Feature1', 'Feature2', 'Feature3' (independientes) y 'Target' (dependiente)
X = data[['Feature1', 'Feature2', 'Feature3']]
y = data['Target']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal múltiple
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

# Coeficientes del modelo
print('Coeficientes:', modelo.coef_)
print('Intercepto:', modelo.intercept_)