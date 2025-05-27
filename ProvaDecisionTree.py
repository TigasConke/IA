import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

#Craiando DataFrame
df = pd.read_csv('csv/social-media.csv')

#Mostrando o DataFrame
print(df)

X = df[['TotalLikes', 'Age', 'UsageDuraiton']]
y = df['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Acuracia: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(40,20))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_, filled=True)
plt.show()

novo_dado = pd.DataFrame({'TotalLikes': [7], 'Age': [12], 'UsageDuraiton': [10]})
nova_previsao = modelo.predict(novo_dado)

print("O país previsto é: ", nova_previsao[0])