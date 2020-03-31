import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Lectura dels data sets:
#gold_ds = pd.read_csv('./data/gold.csv')
#palladium_ds = pd.read_csv('./data/palladium.csv')
#platinum_ds = pd.read_csv('./data/platinum.csv')
#silver_ds = pd.read_csv('./data/silver.csv')
gold_ds = pd.read_csv('./data/gld_price_data.csv')

gold_ds.info()
gold_ds.describe()

corr = gold_ds.corr()
plt.figure(figsize = (6, 5))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, fmt='.2f', linewidths=0.30)
plt.title('Correlation of DataFrame Feature', y=1.05, size=15)

print(corr['GLD'].sort_values(ascending=False), '\n')

sns.distplot(gold_ds['GLD'], color='blue')
print('Skewness: %f' % gold_ds['GLD'].skew())
print('Kurtosis: %f' % gold_ds['GLD'].kurt())

sns.jointplot(x =gold_ds['SLV'], y = gold_ds['GLD'], color = 'deeppink')
sns.jointplot(x =gold_ds['SPX'], y = gold_ds['GLD'], color = 'purple')

x_trail = gold_ds[['SPX','USO','SLV','EUR/USD']]
x = x_trail.iloc[:, :].values
y = gold_ds.iloc[:, 2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))
print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

accuracy_train = regressor.score(x_train, y_train)
accuracy_test = regressor.score(x_test, y_test)
print(accuracy_train)
print(accuracy_test)

plt.plot(y_test, color = 'blue', label = 'Acutal')
plt.plot(y_pred, color = 'deeppink', label = 'Predicted')
plt.grid(0.3)
plt.title('Acutal vs Predicted')
plt.xlabel('Number of Oberservation')
plt.ylabel('GLD')
plt.legend()
plt.show()
