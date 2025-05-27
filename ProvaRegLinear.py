import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error

data = {
    'HorasDeEstudo': [5, 10, 15, 20, 25],
    'NotaFinal': [55, 60, 65, 75, 86]
}

df = pd.DataFrame(data)
print(df)

X = df[['HorasDeEstudo']]
y = df[['NotaFinal']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"MSE = {mse}" )
print(f"Coeficiente = ", modelo.coef_)
print(f"Intercepto =", modelo.intercept_)

nova_entrada = np.array([[17]])
nova_previsao = modelo.predict(nova_entrada)

print("A nota previsata é", nova_previsao)


