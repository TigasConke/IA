# Bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados de exemplo com múltiplas entradas
data = {
    'tempo_estudo': [2, 4, 6, 8, 10, 12],  # entrada 1
    'frequencia': [50, 60, 70, 80, 90, 100],  # entrada 2
    'nota_final': [5, 6, 7, 8, 9, 10]  # saída
}

# DataFrame
dados = pd.DataFrame(data)

# Entradas (duas colunas)
X = dados[['tempo_estudo', 'frequencia']]
# Saída
y = dados['nota_final']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Criação e treino do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsão
y_pred = modelo.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")

# Coeficientes e intercepto
print("Coeficientes (para cada entrada):", modelo.coef_)
print("Intercepto:", modelo.intercept_)

# Nova entrada: 7 horas de estudo e 85% de frequência
nova_entrada = np.array([[7, 85]])

# Previsão com o modelo treinado
nova_previsao = modelo.predict(nova_entrada)

print(f"Nota prevista: {nova_previsao[0]:.2f}")

