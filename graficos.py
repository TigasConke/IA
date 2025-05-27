import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('csv/student-mat.csv', sep=';')

print("Colunas disponíveis:")
print(df.columns.tolist())

#Disperção Horas de estudo X Nota final
plt.scatter(df['studytime'], df['G3'])
plt.title("Horas de estudo X Nota final")
plt.xlabel("Horas de estudo")
plt.ylabel("Nota Final")

plt.show()

#Boxplot Nota final X Consumo de Alcool
plt.figure(figsize=(8, 5))
sns.boxplot( x = 'Dalc', y = 'G3', data = df)
plt.title("Desempenho por consumo de alcool")
plt.ylabel("Nota final")
plt.xlabel("Consumo de alcool")
plt.show()
