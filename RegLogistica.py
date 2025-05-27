# Importação das bibliotecas necessárias
import numpy as np  # Biblioteca para operações matemáticas e manipulação de arrays
import pandas as pd  # Biblioteca para criar e manipular tabelas de dados
from sklearn.model_selection import train_test_split  # Função para dividir os dados em treino e teste
from sklearn.linear_model import LogisticRegression  # Algoritmo de regressão logística
from sklearn.metrics import accuracy_score  # Função para calcular a acurácia do modelo

# Dados fictícios (horas de estudo vs aprovação)
data = {  # Criamos um dicionário com horas de estudo e se passou ou não
    'hours': [1, 2, 3, 4, 5, 6],  # Horas de estudo
    'pass': [0, 0, 0, 1, 1, 1]  # 0 = Reprovado, 1 = Aprovado
}

# Convertendo para DataFrame
study_data = pd.DataFrame(data)  # Transformamos o dicionário em um DataFrame

# Separando variáveis independentes (X) e dependente (y)
X = study_data[['hours']]  # Selecionamos "hours" como variável independente
y = study_data['pass']  # Selecionamos "pass" como variável dependente

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% dos dados para teste

# Criando o modelo de Regressão Logística

model = LogisticRegression()  # Instanciamos o modelo

# Treinando o modelo
model.fit(X_train, y_train)  # Ajustamos o modelo aos dados de treino

# Fazendo previsões
predictions = model.predict(X_test)  # Geramos as previsões para os dados de teste
print(f"Previsões: {predictions}")
print(X_test)

# Avaliando o modelo
accuracy = model.score(X_test, y_test)  # Calculamos a acurácia do modelo
print(f"Acurácia: {accuracy}")