import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

#Craiando DataFrame
df = pd.read_csv('csv/Social_Network_Ads.csv')

#Mostrando o DataFrame
print(df)

#Separando atributos e rótulos
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

#Dividindo entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Criando e treinando o modelo
modelo = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)
modelo.fit(X_train, y_train)

#Prevendo no conjunto de teste
y_pred = modelo.predict(X_test)

#Avaliando modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Visaulização grafica da árvore
plt.figure(figsize=(8,6))
plot_tree(modelo, feature_names=X.columns, class_names=[str(cls) for cls in modelo.classes_], filled=True)
plt.show()

#Criando novo dado como DataFrame
dado_novo = pd.DataFrame({'Gender': [1], 'Age': [60], 'EstimatedSalary': [10000]})

# Aplicando get_dummies para garantir as mesmas colunas
dado_novo = pd.get_dummies(dado_novo, columns=['Gender'], drop_first=True)

# Garantindo que todas as colunas do treino estejam presentes no novo dado
for col in X.columns:
    if col not in dado_novo.columns:
        dado_novo[col] = 0  # adiciona coluna faltante com valor 0

# Reordenando as colunas para garantir a ordem correta
dado_novo = dado_novo[X.columns]

# Fazendo a previsão
nova_previsao = modelo.predict(dado_novo)

# Printando o resultado da previsão
print("Comprou (1) ou não comprou (0): ", nova_previsao[0])

