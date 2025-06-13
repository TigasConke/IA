import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/TigasConke/IA/refs/heads/main/medical_costs.csv"
df = pd.read_csv(url)

# Mapeamento (atribuindo valores numéricos para sexo, regiões e fumante ou não fumante)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Region'] = df['Region'].map({
    'southwest': 0,
    'southeast': 1,
    'northwest': 2,
    'northeast': 3
})

df['Smoker'] = df['Smoker'].map({'no': 0, 'yes': 1})

print(df)

# Verificando valores ausentes
print(df.isnull().sum())

# Removendo linhas com valores ausentes
df = df.dropna()

# Removendo linhas duplicadas
df = df.drop_duplicates()

# Histograma dos custos médicos
plt.hist(df['Medical Cost'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribuição dos Custos Médicos')
plt.xlabel('Custo Médico')
plt.ylabel('Frequência')
plt.show()


# Gráfico de setores das regiões
regioes_labels = ['Southwest', 'Southeast', 'Northwest', 'Northeast']
regioes_counts = df['Region'].value_counts().sort_index()

plt.figure(figsize=(6, 6))
plt.pie(regioes_counts, labels=regioes_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
plt.title('Proporção de Regiões')
plt.show()

# Gráfico de barras sobre o impacto do sexo no custo médico
genero_labels = ['Masculino', 'Feminino']
custo_medio_genero = df.groupby('Sex')['Medical Cost'].mean()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='Medical Cost', data=df)
plt.title('Custo Médico por Sexo')
plt.xlabel('Sexo (0 = Masculino, 1 = Feminino)')
plt.ylabel('Custo Médico')
plt.show()

# Gráfico de barras sobre o impacto do status de fumante no custo médico
plt.figure(figsize=(8, 5))
sns.boxplot(x='Smoker', y='Medical Cost', data=df)
plt.title('Custo Médico por Status de Fumante')
plt.xlabel('Fumante (0 = Não, 1 = Sim)')
plt.ylabel('Custo Médico')
plt.show()

# Gráfico de barras sobre impacto da quantidade de filhos no custo médico
plt.figure(figsize=(8, 5))
cores = ['skyblue', 'orange', 'green', 'red', 'purple', 'gold']
custo_medio_filhos = df.groupby('Children')['Medical Cost'].mean()
plt.bar(custo_medio_filhos.index, custo_medio_filhos.values, color=cores[:len(custo_medio_filhos)], edgecolor='black')
plt.title('Custo Médico Médio por Quantidade de Filhos')
plt.xlabel('Quantidade de Filhos')
plt.ylabel('Custo Médico Médio')
plt.xticks(custo_medio_filhos.index)
plt.show()

# Gráfico de barras sobre custo médico médio por região
plt.figure(figsize=(8, 5))
custo_medio_regiao = df.groupby('Region')['Medical Cost'].mean()
plt.bar(regioes_labels, custo_medio_regiao.values, color=plt.cm.Pastel1.colors[:len(custo_medio_regiao)], edgecolor='black')
plt.title('Custo Médico Médio por Região')
plt.xlabel('Região')
plt.ylabel('Custo Médico Médio')
plt.show()

# Mapa de calor para visualizar quais features são mais correlatas com o custo médico
plt.figure(figsize=(10, 7))
correlacao = df.corr(numeric_only=True)
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações entre as features')
plt.show()

X = df[['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']]
y = df['Medical Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")
print("Coeficientes (para cada entrada):", modelo.coef_)
print("Intercepto:", modelo.intercept_)

nova_entrada = pd.DataFrame([[21, 0, 19.14, 0, 0, 1]], columns=['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region'])
nova_previsao = modelo.predict(nova_entrada)
print(f"Custo médico previsto: ${nova_previsao[0]:.2f}")
# Dados da nova entrada: 21 anos, masculino, IMC 19.14, 0 filhos, não fumante, região sudeste