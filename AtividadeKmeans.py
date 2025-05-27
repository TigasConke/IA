import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#definindo o dataframe
data = pd.read_csv('csv/Mall_Customers.csv')

#printando principais do dataframe
print(data)

#atribuindo a X as entradas
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#Apicando o metodo do cotovelo
inercia = []

#teste de k de 1 a 10

for k in range(1,11):
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X)
    inercia.append(modelo.inertia_)

#grafico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1,11), inercia, marker='o')

#exibindo o grafico do cotovelo
plt.show()

#silhouette_score
sil_scores = []
for k in range(2,11):
    modelo = KMeans(n_clusters=k, random_state=42)
    labels = modelo.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)
plt.plot(range(2,11), sil_scores, marker='o')

#mostrando silhouette_score
plt.show()


#com base no grafico do cotovelo, escolhi k=4 que é onde a inercia em relação a k nao muda tao abruptamente

#Criando o modelo KMeans com os 4 clusters
modelo = KMeans(n_clusters=4, random_state=42)

#Treinando o modelo e colocando em uma nova coluna o resultado
data['cluster'] = modelo.fit_predict(X)

#Mostrando os centroides
print("Centroides:")
print(modelo.cluster_centers_)

#cluster de cada cliente printado
print(data)

#Gráfico dos clusters
plt.figure(figsize=(8,6))

#Colorindo para ficar mais visual
plt.scatter(data['Age'], data['Spending Score (1-100)'], c=data['cluster'], cmap='viridis', s=100)

#mostrando o grafico KMeans
plt.show()

#Interpretação do gráfico: Os clientes entre 18 anos até 40 anos tendem a ter os scores mais altos