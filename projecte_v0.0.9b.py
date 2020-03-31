import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv') #), na_values='nul')
print(dataset.head())
print(dataset.dtypes)
print(dataset.info())

dataset['SILVER'] = dataset['SILVER'].replace({'nan': np.nan})
dataset['SILVER'] = dataset['SILVER'].replace({'nul': np.nan})
dataset['SILVER'] = dataset['SILVER'].astype(float)
print(dataset.head())
print(dataset.dtypes)
print(dataset.info())
print(dataset.isnull().sum())

dataset = dataset.dropna(how='all')
print(dataset.isnull().sum())

for column in dataset:
    if (column != 'Date' and column != 'GOLD'):
        print('Column name: ', column)
        dataset[column] = dataset[column].fillna(dataset[column].mean())
    
print(dataset.isnull().sum())




dataset['GOLD'] = dataset['GOLD'].astype('category')
