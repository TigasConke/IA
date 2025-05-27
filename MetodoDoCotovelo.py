import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dados = {
    'cliente_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'tempo_na_loja': [10, 45, 30, 20, 60, 15, 50, 40, 25, 35],  # em minutos
    'valor_medio_compra': [20, 200, 150, 50, 300, 25, 250, 180, 60, 100],  # em reais
    'visitas_mensais': [1, 5, 4, 2, 6, 1, 5, 4, 2, 3]  # visitas por mês
}
df = pd.DataFrame(dados)

X = df[['tempo_na_loja', 'valor_medio_compra']]

# Aplicando o método do cotovelo
inercia = []

# Testando valores de k de 1 a 10
for k in range(1, 11):
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X)
    inercia.append(modelo.inertia_)

# Plotando o gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inercia, marker='o')
# plt.title('Método do Cotovelo')
# plt.xlabel('Número de Clusters (k)')
# plt.ylabel('Inércia')
# plt.grid(True)
plt.show()