#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#training the multiple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#prediciting the test result
y_pred = regressor.predict(X_test)


#evaluating the performance of the regression model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
