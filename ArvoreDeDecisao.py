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

# Dados com nomes de colunas
data = {
    'Peso': [150, 170, 140, 130, 155, 135, 165, 145],
    'Textura': [1, 0, 1, 0, 1, 0, 0, 1],
    'Fruta': ['maçã', 'laranja', 'maçã', 'laranja', 'maçã', 'laranja', 'laranja', 'maçã']
}

# Criar DataFrame
df = pd.DataFrame(data)

# Separar atributos (X) e rótulos (y)
X = df[['Peso', 'Textura']]
y = df['Fruta']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)
modelo.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = modelo.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualizar a árvore
plt.figure(figsize=(8,6))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_, filled=True)
plt.show()


# Criar novo dado como DataFrame
novo_dado = pd.DataFrame({'Peso': [160], 'Textura': [1]})

# Fazer a previsão
nova_predicao = modelo.predict(novo_dado)

# Mostrar o resultado
print("Previsão para o novo dado:", nova_predicao[0])
