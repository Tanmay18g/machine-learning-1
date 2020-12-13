#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


#training the support vector regression model on training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


#prediciting the test result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))


#evaluating the performance of the regression model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
