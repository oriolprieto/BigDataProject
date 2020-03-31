import pandas as pd
import numpy as np

dataSets = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6], columns=['Name', 'sName', 'DataFrame', 'Rows'])
dataSets.at[0, 'Name'] = "Dow Jones"
dataSets.at[0, 'sName'] = "DJI"
dataSets.at[0, 'DataFrame'] = pd.read_csv('^DJI.csv')
dataSets.at[1, 'Name'] = "Standard & Poors"
dataSets.at[1, 'sName'] = "SP00"
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
for index in dataSets.index: dataSets.at[index, 'Rows'] = 0
dataSets['Rows'] = dataSets['Rows'].astype(int)
print(dataSets.dtypes)
print(dataSets.info())

for name, df, index in zip(dataSets.iloc[:, 0], dataSets.iloc[:, 2], dataSets.index): 
    print('\n********************************************\nProcesing data set: ' + name)
    print(df.info())
    df = df[['Date', 'Close']]
    df = df.replace({'nan', np.nan})
    df = df.replace({'nul', np.nan})
    print('Before drp NaN\n', df.isnull().sum())
    df = df.dropna(how='any')
    print('After drop NaN\n', df.isnull().sum())
    print(df.info(), "\nCounts:\n", df.count())
    dataSets.at[index, 'Rows'] = df['Close'].count()
    df = df.set_index('Date')
    dataSets.at[index, 'DataFrame'] = df
    
dataset = pd.DataFrame()
for index, df in zip(dataSets.index, dataSets.iloc[:,2]):
    df = df.rename(columns={"Close": dataSets.at[index, 'sName']})
    dataset = pd.concat([dataframe, df], axis=1, sort=True)



print(df.iloc[0,0])
print(df['Date'].count())
print(dataSets['Rows'].max())
print(dataSets['Rows'].idxmax())
print(dataSets.info())    



dowJonesDataset = pd.read_csv('^DJI.csv')
sPoorsDataset = pd.read_csv('^GSPC.csv')
ibex35Dataset = pd.read_csv('^IBEX.csv')
crudeDataset = pd.read_csv('CL=F.csv')
goldDataset = pd.read_csv('GC=F.csv')
silverDataset = pd.read_csv('SI=F.csv')
eurUsdDataset = pd.read_csv('EURUSD=X.csv')

dfList = [
        ["Dow Jones", dowJonesDataset],
        ["Standard & Poors", sPoorsDataset],
        []



for datasetInList in  dataSets:
    datasetInList = datasetInList[['Date', 'Close']]
    datasetInList = datasetInList.replace({'nan', np.nan})
    datasetInList = datasetInList.replace({'nul', np.nan})
    print(datasetInList.isnull().sum())
    datasetInList = datasetInList.dropna(how='any')
    print(datasetInList.isnull().sum())



dowJonesDataset = dowJonesDataset[['Date', 'Close']]
dowJonesDataset = dowJonesDataset.replace({'nan', np.nan})
dowJonesDataset = dowJonesDataset.replace({'nul', np.nan})
print(dowJonesDataset.isnull().sum())
dowJonesDataset = dowJonesDataset.dropna(how='all')





dataset = pd.read_csv('dataset.csv') #), na_values='nul')
print(dataset.head())
print(dataset.dtypes)
print(dataset.info())

dataset['GOLD'] = dataset['GOLD'].astype('category')
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
