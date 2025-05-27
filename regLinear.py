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

# Imports para visualização de dados (opcional, mas recomendado)
import matplotlib.pyplot as plt
import seaborn as sns

# Gerando dados de exemplo (área da casa vs preço)
data = {  # Criamos um dicionário onde cada chave é uma variável
    'area': [50, 60, 70, 80, 90, 100],  # Área da casa em metros quadrados
    'price': [150, 180, 210, 240, 270, 300]  # Preço em milhares de reais
}

# Convertendo para DataFrame
house_data = pd.DataFrame(data)  # Transformamos o dicionário em uma tabela para facilitar a manipulação

print(house_data)
# Dividindo as variáveis independentes (X) e dependente (y)
X = house_data[['area']]  # Selecionamos apenas a coluna "area" como variável independente
y = house_data['price']   # Selecionamos a coluna "price" como variável dependente

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% treino, 20% teste

# Criando o modelo de Regressão Linear
model = LinearRegression()  # Instanciamos o modelo

# Treinando o modelo
model.fit(X_train, y_train)  # Ajustamos o modelo aos dados de treino

# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)  # Geramos as previsões para os dados de teste

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)  # Calculamos o erro quadrático médio
print(f"Erro Quadrático Médio: {mse}")
print(X_test)
print(y_pred)

# Exibindo o coeficiente angular e o intercepto
print(f"Coeficiente (pendente): {model.coef_[0]}")  # Inclinação da reta de regressão
# print("Coef: ", model.coef_[0])
print(f"Intercepto: {model.intercept_}")  # Valor de interseção com o eixo y