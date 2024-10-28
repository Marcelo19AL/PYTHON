import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Cargar el dataset
dataset = pd.read_csv("csv/50_Startups.csv")
print("Datos cargados correctamente:")
print(dataset.head())  # Muestra las primeras filas del dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4]

# Preprocesamiento
le_X = preprocessing.LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])

ct = ColumnTransformer([('One_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float64)

X = X[:, 1:]

# Dividir los datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Ajustar el modelo de regresión lineal
regression = LinearRegression()
regression.fit(X_train, Y_train)

Y_pred = regression.predict(X_test)

# OLS
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# Ajustar el modelo OLS
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(Y, X_opt).fit()
print(regression_OLS.summary())  # Asegúrate de que esto se ejecute
