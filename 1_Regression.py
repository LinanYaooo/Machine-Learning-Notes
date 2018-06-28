# Machine Learning Tutorial - 1. Regression
# 1. Sklearn 2.Quandl 3.Pandas
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
df = quandl.get('WIKI/GOOGL')
print(df)
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
df.dropna(inplace=True)

# 3
x = np.array(df.drop(['label'], 1))                # Allow arrary in Python;
y = np.array(df['label'])                         # Arrarilise a dataset;
x = preprocessing.scale(x)                        # Step size ? Weighting ?

# Linear Regressior
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)                 # Fit a linear classfier;
clf.fit(x_train, y_train)                         # Fit the estimator;
accuracy = clf.score(x_test, y_test)              # Test the estimator;
print(accuracy)
# Support Vector Regressior Machine
clf2 = svm.SVR()
clf2.fit(x_train, y_train)
accuracy2 = clf2.score(x_test, y_test)
print(accuracy2)

# 4
