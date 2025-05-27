import pandas as pd

# Criação de um Dicionário chamado data que contém dados simulados
data = {
    # Coluna Nome: Vale notar que tem um valor ausente (None) nessa coluna
    'Nome' : ['Tiago', 'Marta', 'Julio', 'Marcela', None, 'Tiago'],
    # Coluna Idade: Vale notar que tem um valor ausente (None) nessa coluna
    'Idade' : [21, 12, 25, None, 79 , 21],
    # Coluna Salário: Vale notar que tem um valor ausente (None) nessa coluna
    'Salário' : [2166, 4200, 5666.90, None, 79000 , 2166],
    # Coluna Cidade
    'Cidade' : ['Ponta Grossa', 'São Paulo', 'Foz do Iguaçu', 'Salvador', 'Maceió' , 'Ponta Grossa'],
}

#Criando um DataFrame (tabela de dados) com base no dicionário que criei
df = pd.DataFrame(data)

#Exibição do DataFrame
print("Dados Originais:")
print(df)

print("\nVerificando valores ausentes:")
print(df.isnull())
print("\nQuantidade de vlaores ausentes por coluna:")
print(df.isnull().sum())

#Limpeza de linhas com valores ausentes (null)
df_cleaned = df.dropna()
print("\nDados após limpeza de linhas com valores ausente (null):")
print(df_cleaned)

#preenchendo valores ausentes na coluna Idade com a média
df['Idade'] = pd.to_numeric(df['Idade'], errors='coerce')  # converte para número e coloca NaN onde falhar
df['Idade'] = df['Idade'].fillna(df['Idade'].mean())       # agora sim, preenche os NaN com a média
print("\nDados após preencher valores ausentes em Idade com a média:")
print(df)

#Verificando e exibindo duplicatas
print("\nLinhas duplicadas:")
print(df.duplicated())

#Removendo duplicatas
df = df.drop_duplicates()
print("\nDados após remover duplicatas:")
print(df)

# Padronizando os nomes das cidades
# Converte todos os nomes para letras minúsculas (str.lower())
df['Cidade'] = df['Cidade'].str.lower()

# Exibindo os dados padronizados
print("\nDados com padronização de cidades:")
print(df)



# Ler o arquivo Excel (substitua 'arquivo.xlsx' pelo nome do arquivo)
# df2 = pd.read_excel('arquivo.xlsx')

# Exibir os primeiros dados
# print(df2.head())



# Outras manipulações de dados com pandas
# **1. Criando uma nova coluna com base em cálculos**
# Criar uma coluna "Salário Anual" multiplicando o salário mensal por 12
df['Salário Anual'] = df['Salário'] * 12
# **2. Filtragem de dados**
# Filtrar as linhas onde a idade é maior que 25
df_filtrado = df[df['Idade'] > 25]
# **3. Reordenando colunas**
# Alterar a ordem das colunas para: Nome, Cidade, Idade, Salário, Salário Anual
df = df[['Nome', 'Cidade', 'Idade', 'Salário', 'Salário Anual']]
# Exibindo os resultados
print("\nDataFrame após criar 'Salário Anual':")
print(df)
print("\nDataFrame filtrado (Idade > 25):")
print(df_filtrado)
