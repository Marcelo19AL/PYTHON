# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
file_path = 'csv/Fast_Food_Restaurants.csv'  # Asegúrate de que la ruta sea correcta
df = pd.read_csv(file_path)

# Mostrar los primeros registros del conjunto de datos
print("Datos cargados correctamente:")
print(df.head())

# Resumen del conjunto de datos
print("\nResumen del conjunto de datos:")
print(df.info())

# Verificar si hay valores nulos
print("\nValores nulos en cada columna:")
print(df.isnull().sum())

# Visualizar la distribución de restaurantes por estado
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='province', order=df['province'].value_counts().index, palette='viridis')
plt.title('Número de Restaurantes de Comida Rápida por Estado')
plt.xlabel('Estado')
plt.ylabel('Número de Restaurantes')
plt.xticks(rotation=45)
plt.show()

# Visualizar la relación entre la cantidad de restaurantes y la población
# Supongamos que tenemos una columna 'population' que contiene la población de cada ciudad
# Aquí debes agregar una lógica para agregar una columna de población si la tienes.
# Si no, puedes comentar esta parte.

# df['restaurants_per_capita'] = df['number_of_restaurants'] / df['population']

# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='population', y='number_of_restaurants', hue='province', palette='deep')
# plt.title('Número de Restaurantes de Comida Rápida por Población')
# plt.xlabel('Población')
# plt.ylabel('Número de Restaurantes')
# plt.legend(title='Estado')
# plt.show()

# Análisis de las categorías de restaurantes
plt.figure(figsize=(12, 6))
category_counts = df['categories'].value_counts().head(10)
sns.barplot(x=category_counts.values, y=category_counts.index, palette='magma')
plt.title('Top 10 Categorías de Restaurantes de Comida Rápida')
plt.xlabel('Número de Restaurantes')
plt.ylabel('Categorías')
plt.show()

# Guardar el DataFrame limpio (si es necesario)
# df.to_csv('Cleaned_Fast_Food_Restaurants.csv', index=False)

print("Análisis completo.")
