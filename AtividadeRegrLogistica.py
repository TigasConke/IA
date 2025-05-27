# Importação das bibliotecas necessárias
import numpy as np  # Biblioteca para operações matemáticas e manipulação de arrays
import pandas as pd  # Biblioteca para criar e manipular tabelas de dados
from sklearn.model_selection import train_test_split  # Função para dividir os dados em treino e teste
from sklearn.linear_model import LogisticRegression  # Algoritmo de regressão logística
from sklearn.metrics import accuracy_score  # Função para calcular a acurácia do modelo

#Idade X tem cancer ou nao
dados = {
    'idade': [20, 30, 40, 50, 60, 70, 80, 21, 24, 56],
    'cancer': [0, 0, 1, 0, 1, 1, 0, 1, 1, 0], #1 = tem cancer e 0 = não tem cancer
}

cancer_data = pd.DataFrame(dados)
print(cancer_data)

X = cancer_data[['idade']]
y = cancer_data['cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

previsoes = model.predict(X_test)

print(f"Previsões: {previsoes}")
print(X_test)

accuracy = model.score(X_test, y_test)
print(f"Acurácia: {accuracy}")

nova_entrada = np.array([[16]])  # 21 anos
nova_previsao = model.predict(nova_entrada)
print(f"Previsão para uma pessoa com {nova_entrada[0]} anos ter ou não câncer: {nova_previsao[0]}")
