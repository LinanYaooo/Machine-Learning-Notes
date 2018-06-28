# Use pickle to save a classfier
# Machine Learning Tutorial - 1. Regression
# 1. Sklearn 2.Quandl 3.Pandas
import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt                             # Plotting
from matplotlib import style                                # Make things decent
import pickle

style.use('ggplot')                                         # How decent you want
df = quandl.get('WIKI/GOOGL')
# print(df)
# 1 Define some features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # Grab some features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * \
    100.0  # Calculate the High-Low Percentage CHange
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * \
    100.0  # Percentage CHange daily
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# 2 Define a label
forecast_col = 'Adj. Close'                       # To forecast the Closed price of stock;
df.fillna(-99999, inplace=True)                   # IfNull(), replace with -99999;
forecast_out = int(math.ceil(0.01 * len(df)))     # 0.01 dateframe
# Define a label, real close data of 0.01 dateframe after
df['label'] = df[forecast_col].shift(-forecast_out)

# 3
x = np.array(df.drop(['label'], 1))                # Allow arrary in Python;
x = preprocessing.scale(x)                         # Step size ? Weighting ?
x = x[:-forecast_out]                              # All data except the last 30 days
x_lately = x[-forecast_out:]                       # Save the data of last 30 days in another place


df.dropna(inplace=True)                            # Drop null value;
y = np.array(df['label'])                          # Arrarilise a dataset;


# Linear Regressior
# x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
# clf = LinearRegression(n_jobs=-1)                  # Fit a linear classfier;
# clf.fit(x_train, y_train)                          # Fit the estimator, Training Step;
#
# # Saving the classfier, not have to re-train the classfier
# with open('LinearRegression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
pickle_in = open('LinearRegression.pickle', 'rb')
clf = pickle.load(pickle_in)

# 4 Predict based on x data
# use the first classfier to predict the close from close in 30 before;
forecast_set = clf.predict(x_lately)
# Print out the forecasted close and accuracy and the length of date


df['Forecast'] = np.nan                            # Create a new column;
last_date = df.iloc[-1].name                       # take out the index of the last date;
last_unix = last_date.timestamp()                  # Get the timestamp of the last date;
one_day = 86400                                    # How mach second daily;
next_unix = last_unix + one_day                    # Create time axis

for i in forecast_set:                                           # for each prediction
    next_date = datetime.datetime.fromtimestamp(next_unix)       # Generate time axis
    next_unix += one_day                                         # Update the date
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print(df)
