import numpy as np  # Biblioteca para manipulação de arrays e funções matemáticas
import pandas as pd  # Biblioteca para manipulação de dados em formato tabular
from sklearn.model_selection import train_test_split  # Função para dividir os dados em treino e teste
from sklearn.linear_model import LinearRegression  # Modelo de regressão linear
from sklearn.metrics import mean_squared_error  # Métrica para avaliar o modelo

data = {
    'Horas de Estudo' : [2, 3, 6, 12, 5, 7],
    'Nota' : [5, 5, 8, 9, 7, 8]
}

desempenho = pd.DataFrame(data)
print(desempenho) #exibindo o dataframe

X = desempenho[['Horas de Estudo']]
y = desempenho[['Nota']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% treino, 20% teste

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")
print(X_test)
print(y_pred)

print(f"Coeficiente (pendente): {model.coef_[0]}")
print(f"Intercepto: {model.intercept_}")

nova_entrada = np.array([[10]])  # 10 horas de estudo
nova_previsao = model.predict(nova_entrada)
print(f"Nota prevista para 10 horas de estudo: {nova_previsao[0][0]:.2f}")
