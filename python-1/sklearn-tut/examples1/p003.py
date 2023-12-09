import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("darkgrid")

# Load the iris dataset
iris = datasets.load_iris()
print("Description: ", iris.DESCR)

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.describe())
print(df.corr())

y = df['petal width (cm)']
x = df[['petal length (cm)', 'sepal length (cm)']]

print(x.shape)
print(y.shape)

# Split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print('R2 score: \n', regr.score(x_test, y_test))

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# print the first 5 predictions
print(x_test[:5])
print(y_pred[:10])

# mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
