import math
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = 'yapVTe9oroDbDJEhs3Rp'

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL%'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low']*100.0
df['CHANGE%'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL%', 'CHANGE%', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.1*len(df)))    # no of days after present day for which we want to predict stock price
print(forecast_out)                                            # currently set to 1% of total number of days for which we have data
df['Label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)
print(df.head())

X = np.array(df.drop(['Label'], 1))

X = preprocessing.scale(X)
X_recent = X[-forecast_out:]
# print(len(X))
X= X[:-forecast_out]



df.dropna(inplace=True)     # since we shifted the column Label up by forecast_out, we have forecast_out amount of NA values which will be dropped here
y = np.array(df['Label'])

# df.dropna(inplace=True)
# print(len(X), len(y))
# print(X)
# print(y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

classifier = LinearRegression()
# classifier = svm.SVR(kernel = 'poly')   # pretty bad accuracy around 67%
classifier.fit(X_train, y_train)
with open('LR.pickle', 'wb') as f:
    pickle.dump(classifier, f)
pickle_in = open('LR.pickle', 'rb')
classifier = pickle.load(pickle_in)
accuracy = classifier.score(X_test, y_test)
forecast_set = classifier.predict(X_recent)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + 86400   # next day, adding 86,400 seconds to last timestamp

# print(last_unix)

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=86400
    print(next_date, next_unix)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# print(df.head)
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


