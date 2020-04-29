import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA,ARIMA


import warnings
import time

################################################################################################
# 1. Llegir dades i construir dataset.
################################################################################################
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
#Posem l'index com a datetime
dataset.index = pd.to_datetime(dataset.index) 

################################################################################################
# 2. Analitzem la correlació entre indicadors
################################################################################################
#Obtenim la matriu de correlació de tots amb tots
dataset_corr = dataset.corr()
#print(dataset_corr)

#Obtenim la correlacio amb l'or
print('GOLD Correlation with other parameters:')
dataset_corr_values = dataset_corr['GOLD'][:-1].abs().sort_values(ascending=False)
print(dataset_corr_values)

#Representem el valor absolut de la correlacio
print('\n')
graphic=sns.heatmap(dataset_corr.abs(),annot=True,linewidths=0.5,vmin=0, vmax=1)
graphic.set(title = "Correlation Matrix Absolute Values")
print('\n')

################################################################################################
# 3. Provem clasificadors regresius, i els optimitzem
################################################################################################
#Provem Clasificadors Regresius (per predeir número) que no depenent en el temps 
X = dataset.iloc[:, :-1]
y = dataset['GOLD']

#Fem el split amb dates
X_train = X.loc['2010-01-01':'2017-12-31']
X_test = X.loc['2018-01-01':]
y_train = y.loc['2010-01-01':'2017-12-31']
y_test = y.loc['2018-01-01':]

#Provem els seguents clasificadors: (GNB -> Bayesian Ridge...)
names = ["KNNReg","TreeReg.", "MLPReg.", "SVMReg.","ForestReg."]
classifiers = [
        KNeighborsRegressor(n_neighbors=10),
        DecisionTreeRegressor(max_depth=20),
        MLPRegressor(alpha=1, max_iter=1000),
        SVR(C=1.0, epsilon=0.2),
        RandomForestRegressor(n_estimators = 100, random_state = 0)]

#Comparem les seguents característiques:
results = pd.DataFrame(index=['Absolute Error','Variance Score','Train Cost', 'Test Cost'], columns=names)

for name, clf in zip(names, classifiers):
    t1=time.time()
    clf.fit(X_train, y_train)
    t2=time.time()
    y_pred = clf.predict(X_test)
    t3=time.time()
    results.at['Train Cost', name]=round(t2-t1,3);
    results.at['Test Cost', name]=round(t3-t2,3);
    results.at['Absolute Error', name]=mean_absolute_error(y_test, y_pred);
    results.at['Variance Score', name]=explained_variance_score(y_test, y_pred);
    
print('Results of Regression Classifiers')
print(results);
#Triar clasificador regressiu i optimitzar parametres

################################################################################################
# 4. Provem clasificador Deep Learning KERAS
################################################################################################
#Provem Clasificador Deep Learning 
#Per executar fa falta instalar Tensor Flow
# conda install -c conda-forge tensorflow

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
def baseline_model():
    model = Sequential()
    model.add(Dense(128, input_dim=6, activation='relu')) 
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256,activation='relu')) 
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=30, verbose=1)
t1=time.time()
estimator.fit(X_train, y_train,)
t2=time.time()
y_pred = estimator.predict(X_test)
t3=time.time();
name="Deep Learning"
results.at['Train Cost', name]=round(t2-t1,3);
results.at['Test Cost', name]=round(t3-t2,3);
results.at['Absolute Error', name]=mean_absolute_error(y_test, y_pred);
results.at['Variance Score', name]=explained_variance_score(y_test, y_pred);

print('Results of Deep Learning')
print(results[name]);

#Com varien resultats podriem fer bucle o KFold.
#from sklearn.model_selection import cross_val_score, KFold
#kfold = KFold(n_splits=10)
#results = cross_val_score(estimator, X_train, y_train, cv=kfold)
#print("Results KFold: %.2f (%.2f) MSE" % (results.mean(), results.std()))

################################################################################################
# 5. Provem clasificadors temporals
################################################################################################
#Provem Clasificadors Temporals (ARIMA,SARIMA...)

#Regresions que només predeixen en funció de l'OR (univariate)

warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)    
names = ["AR","ARMA","ARIMA"]
classifiers = [
        AR(y_train),
        ARMA(y_train, order=(4, 3)),
        ARIMA(y_train, order=(1, 1, 1))] 


#Comparem les seguents característiques:
results = pd.DataFrame(index=['AIC','Absolute Error','Variance Score','Train Cost', 'Test Cost',], columns=names)

for name, clf in zip(names, classifiers):
    t1=time.time()
    model_fit= clf.fit(disp=False)
    t2=time.time()
    y_pred = model_fit.predict('2018-01-01', end='2020-03-27')
    t3=time.time()
    results.at['Train Cost', name]=round(t2-t1,3);
    results.at['Test Cost', name]=round(t3-t2,3);
    #Problema que tenim, que les dates de la predicció no cuadren amb les de test.
    #Cuadrem resultats amb el index
    join = pd.concat([y_pred, y_test], axis=1)
    join=join.dropna()
    results.at['Absolute Error', name]=mean_absolute_error(join['GOLD'], join[0]);
    results.at['Variance Score', name]=explained_variance_score(join['GOLD'], join[0]);
    results.at['AIC', name]=model_fit.aic;
    

print('Results of Regression Classifiers')
print(results);

#Es podria analitzar quina percentatge d'estacionalitat te
#from statsmodels.tsa.seasonal import seasonal_decompose
#result = seasonal_decompose(dataset['GOLD'],period=365, model='additive')
#result.plot()

#Ara com no es estacional no te sentit aplica SARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(2, 1, 2, 100))
#from statsmodels.tsa.ar_model import AutoReg
#AutoReg(data, lags = [1, 11, 12])
