import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

clientes = {
    'cliente_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'tempo_na_loja': [10, 45, 30, 20, 60, 15, 50, 40, 25, 35],  # em minutos
    'valor_medio_compra': [20, 200, 150, 50, 300, 25, 250, 180, 60, 100],  # em reais
    'visitas_mensais': [1, 5, 4, 2, 6, 1, 5, 4, 2, 3]  # visitas por mês
}

df = pd.DataFrame(clientes)

X = df[['tempo_na_loja', 'valor_medio_compra', 'visitas_mensais']]

# Criando o modelo KMeans com 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)

# Treinando o modelo e adicionando o resultado como nova coluna
df['cluster'] = kmeans.fit_predict(X)

# Mostrando os centróides dos clusters
print("Centróides dos clusters:")
print(kmeans.cluster_centers_)

# Exibindo a tabela com o cluster de cada cliente
print("\nClientes com seus respectivos clusters:")
print(df)

# Visualização gráfica dos clusters usando duas variáveis
plt.figure(figsize=(8, 6))

# Cada ponto será colorido de acordo com o cluster
plt.scatter(df['tempo_na_loja'], df['valor_medio_compra'],
            c=df['cluster'], cmap='viridis', s=100)

# Marcando os centróides em vermelho
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             c='red', s=200, marker='X', label='Centróides')

# # Títulos e legendas
# plt.title("Clusters de Clientes com K-Means")
# plt.xlabel("Tempo na Loja (min)")
# plt.ylabel("Valor Médio da Compra (R$)")
# plt.legend()
# plt.grid(True)
plt.show()