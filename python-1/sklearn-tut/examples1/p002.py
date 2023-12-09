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

# Print the first 5 rows
print(df.describe())
print(df.corr())

x = df['petal length (cm)'].values.reshape(-1, 1)
y = df['petal width (cm)'].values.reshape(-1, 1)

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

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, marker='o', linewidth=3, color='red')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Petal length vs Petal width')
plt.xticks(())
plt.yticks(())
plt.show()
