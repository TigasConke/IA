# Import para manipulação de dados
import numpy as np
import pandas as pd

# Imports para os modelos de Machine Learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans

# Imports para avaliação de modelos e divisão de dados
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
# Imports para visualização de dados (opcional, mas recomendado)
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'tempo_estudo': [2, 4, 6, 8, 10, 12],  # entrada 1
    'frequencia': [50, 60, 70, 80, 90, 100],  # entrada 2
    'nota_final': [5, 6, 7, 8, 9, 10]  # saída
}

df = pd.DataFrame(data)

print(df)

X = df[['tempo_estudo', 'frequencia']]
y = df['nota_final']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

previsao = modelo.predict(X_test)

mse = mean_squared_error(y_test, previsao)
print("Mse: ", mse)
print("Coef: ", modelo.coef_)
print("Intercepto: ", modelo.intercept_)

nova_entrada = np.array([[4, 89]])
nova_previsao = modelo.predict(nova_entrada)
print("Nova previsao: ", nova_previsao)


