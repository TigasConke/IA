import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Criação do DataFrame
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

dados = pd.read_csv(url)

#printando principais linhas do dataframe
print(dados)

#Entradas
X = dados[['TV', 'radio', 'newspaper']]

#Saida
y = dados['sales']

#Dividindo dados de teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Criação e treino do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#Previsao
y_pred = modelo.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Coeficientes e intercepto
print("Coeficientes (para cada entrada): ", modelo.coef_)
print("Intercepto: ", modelo.intercept_)

# Previsao de vendas com as novas entradas
novas_entradas = np.array([[123, 6, 49]])
nova_previsao = modelo.predict(novas_entradas)
print(f"Vendas previstas: {nova_previsao[0]}")



