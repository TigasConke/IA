import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#definindo o dataframe
df = pd.read_csv('csv/Mall_Customers.csv')

#printando principais do dataframe
print(df)

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

inercia = []

for k in range(1, 11):
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X)
    inercia.append(modelo.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1,11), inercia, marker='o')
plt.show()

modelo = KMeans(n_clusters=4, random_state=42)

df['cluster'] = modelo.fit_predict(X)

print(modelo.cluster_centers_)

print(df)

plt.figure(figsize=(8,6))

plt.scatter(df['Age'], df['Spending Score (1-100)'], c=df['cluster'], cmap='viridis', s=100)
plt.show()
