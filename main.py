import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Carregando os dados
data = sm.datasets.get_rdataset("AirPassengers").data

# Convertendo 'time' (ano + fração) em datetime com ano e mês corretos
data['Year'] = data['time'].astype(int)
data['Month'] = ((data['time'] - data['Year']) * 12 + 1).round().astype(int)
data['Date'] = pd.to_datetime(dict(year=data['Year'], month=data['Month'], day=1))

# Organizando
data.set_index('Date', inplace=True)
data = data.rename(columns={'value': 'passengers'})
data = data[['passengers']]

# Pronto! Agora sim é uma série temporal com índice mensal
print(data)

plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Série Temporal de Passageiros/Ano')
plt.xlabel('Data')
plt.ylabel('Número de Passageiros')
plt.show()

train = data[:'1959-12']
test = data['1960-01':]

print("Treino:")
print(train.tail())

print("\nTeste:")
print(test.head())

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

forecast = model_fit.get_forecast(steps=len(test))
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(train, label='Treino')
plt.plot(test, label='Teste', color='gray')
plt.plot(predicted_mean, label='Previsão', color='red')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3)
plt.legend()
plt.title('Previsão SARIMA vs Dados Reais')
plt.show()
