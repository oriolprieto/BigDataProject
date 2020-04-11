import pandas as pd
import numpy as np
import seaborn as sns

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

#Mostrem informació de tots els datasets
print(dataSets.head())
print(dataSets.dtypes)
print(dataSets.info())

#Ens quedem amb les columes de Date i Close
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
    
#Concatenem en un dataset, i renombrem les columnes amb el nom curt.
dataset = pd.DataFrame()
for index, df in zip(dataSets.index, dataSets.iloc[:,2]):
    df = df.rename(columns={"Close": dataSets.at[index, 'sName']})
    dataset = pd.concat([dataset, df], axis=1, sort=True)

#Mostrem informació del Dataset abans del preprocessat, inclós el nombre de NaNs
print(dataset.head())
print(dataset.info())
print(dataset.isnull().sum())

#Eliminem les files amb tots NaNs, i mostrem nombre de NaNs que queden
dataset = dataset.dropna(how='all')
print(dataset.isnull().sum())

#Substituim els NaNs pel valor anterior
for column in dataset:
    print('Column name: ', column)
    dataset[column] = dataset[column].fillna(method='bfill')    #fillna(dataset[column].mean())
    
#Mostrem informació del Dataset definitiu, i mostrem que ja no hi ha NaNs
print(dataset.info())
print(dataset.isnull().sum())

#Posem GOLD com a categoria
#dataset['GOLD'] = dataset['GOLD'].astype('category')

#Obtenim la matriu de correlació de tots amb tots
dataset_corr = dataset.corr()
print(dataset_corr)

#Obtenim la correlacio amb l'or
print('GOLD Correlation with other parameters:')
dataset_corr_values = dataset_corr['GOLD'][:-1].abs().sort_values(ascending=False)
print(dataset_corr_values)

#Representem el valor absolut de la correlacio
print('\n')
graphic=sns.heatmap(dataset_corr.abs(),annot=True,linewidths=0.5,vmin=0, vmax=1)
graphic.set(title = "Correlation Matrix Absolute Values")
print('\n')


