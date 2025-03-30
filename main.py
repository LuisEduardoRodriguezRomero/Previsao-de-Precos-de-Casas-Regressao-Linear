import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

n=1000

#Gerando dados ficticios
tamanho = np.random.randint(30,150,n)
quartos = np.random.randint(1,6,n)
banheiros = np.random.randint(1,4,n)
andares = np.random.randint(1,3,n)
idade = np.random.randint(0,50,n)
preco = (
    tamanho * 1000 +
    quartos * 50000 +
    banheiros * 30000 +
    andares * 40000 -
    idade * 2000 +
    np.random.normal(0,50000,n)
).astype(int)

df = pd.DataFrame({
    'tamanho_m2':tamanho,
    'quartos': quartos,
    'banheiros': banheiros,
    'andares': andares,
    'idade_anos': idade,
    'preco_reais': preco
})

print(df.head())

print(df.describe())

print(df.isnull().sum())

sns.histplot(df["preco_reais"],bins=20, kde =True)
plt.show()

corr = df.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm",fmt=".2f")
plt.show()


sns.scatterplot(x=df["tamanho_m2"], y=df["preco_reais"])
plt.show()

sns.boxplot(x=df["quartos"], y=df["preco_reais"])
plt.show()


#Definir features (X) e target (y)
X = df.drop(columns=["preco_reais"]) #Preditoras
y = df["preco_reais"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

#Criando e treinando o modelo

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("Coeficinetes:", modelo.coef_)
print("Interceptor:", modelo.intercept_)


y_pred = modelo.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))