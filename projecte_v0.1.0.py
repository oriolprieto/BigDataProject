import pandas as pd
import numpy as np

dataSets = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6], columns=['Name', 'sName', 'DataFrame'])
dataSets.at[0, 'Name'] = "Dow Jones"
dataSets.at[0, 'sName'] = "DJI"
dataSets.at[0, 'DataFrame'] = pd.read_csv('^DJI.csv')
dataSets.at[1, 'Name'] = "Standard & Poors"
dataSets.at[1, 'sName'] = "SP500"
dataSets.at[1, 'DataFrame'] = pd.read_csv('^GSPC.csv')
dataSets.at[2, 'Name'] = "Ibex-35"
dataSets.at[2, 'sName'] = "IBX35"
dataSets.at[2, 'DataFrame'] = pd.read_csv('^IBEX.csv')
dataSets.at[3, 'Name'] = "Crude"
dataSets.at[3, 'sName'] = "OIL"
dataSets.at[3, 'DataFrame'] = pd.read_csv('CL=F.csv')
dataSets.at[4, 'Name'] = "Silver"
dataSets.at[4, 'sName'] = "SILVER"
dataSets.at[4, 'DataFrame'] = pd.read_csv('SI=F.csv')
dataSets.at[5, 'Name'] = "Eur-Usd"
dataSets.at[5, 'sName'] = "EUR/USD"
dataSets.at[5, 'DataFrame'] = pd.read_csv('EURUSD=X.csv')
dataSets.at[6, 'Name'] = "Gold"
dataSets.at[6, 'sName'] = "GOLD"
dataSets.at[6, 'DataFrame'] = pd.read_csv('GC=F.csv')
print(dataSets.head())
print(dataSets.dtypes)
print(dataSets.info())

for name, df, index in zip(dataSets.iloc[:, 0], dataSets.iloc[:, 2], dataSets.index): 
    print('\n******************************************** Procesing data set: ' + name)
    print(df.info())
    df = df[['Date', 'Close']]
    df = df.replace({'nan', np.nan})
    df = df.replace({'nul', np.nan})
    print('Before drp NaN\n', df.isnull().sum())
    df = df.dropna(how='all')
    print('After drop NaN\n', df.isnull().sum())
    print(df.info(), "\nCounts:\n", df.count())
    df = df.set_index('Date')
    dataSets.at[index, 'DataFrame'] = df
    print('\n++++++++++++++++++++++++++++++++++++++++++++ END for ' + name + '\n')
    
dataset = pd.DataFrame()
for index, df in zip(dataSets.index, dataSets.iloc[:,2]):
    df = df.rename(columns={"Close": dataSets.at[index, 'sName']})
    dataset = pd.concat([dataset, df], axis=1, sort=True)

print(dataset.head())
print(dataset.info())
print(dataset.isnull().sum())

dataset = dataset.dropna(how='all')
print(dataset.isnull().sum())

for column in dataset:
    print('Column name: ', column)
    dataset[column] = dataset[column].fillna(method='bfill')    #fillna(dataset[column].mean())
    
print(dataset.info())
print(dataset.isnull().sum())
dataset['GOLD'] = dataset['GOLD'].astype('category')
