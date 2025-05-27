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

dados = {
    'idade': [20, 30, 40, 50, 60, 70, 80, 21, 24, 56],
    'cancer': [0, 0, 1, 0, 1, 1, 0, 1, 1, 0], #1 = tem cancer e 0 = não tem cancer
}

cancer_data = pd.DataFrame(dados)
print(cancer_data)

X = cancer_data[['idade']]
y = cancer_data['cancer']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

previsao = modelo.predict(X_test)
print(previsao)
print(X_test)
acuracia = modelo.score(X_test, y_test)
print("Acuracia: ", acuracia)

nova_entrada = np.array([[16]])
nova_previsao = modelo.predict(nova_entrada)
print("Nova previsao: ", nova_previsao)