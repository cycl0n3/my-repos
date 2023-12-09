import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from scipy.special import expit

sns.set_style("darkgrid")

# Generate a toy dataset, it's just a straight line with some Gaussian noise:
x_min, x_max = -5, 5
n_samples = 100

np.random.seed(42)

X = np.random.normal(size=n_samples)
y = (X > 0).astype(np.float32)

X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]

# Logistic regression
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

# Linear regression
regr = linear_model.LinearRegression()
regr.fit(X, y)

X_test = np.linspace(-5, 10, 300)

plt.figure(figsize=(12, 8))
plt.scatter(X.ravel(), y, color='black', zorder=20)

# Logistic regression
loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

# Linear regression
ols = X_test * regr.coef_ + regr.intercept_
plt.plot(X_test, ols, color='blue', linewidth=3)

plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(())
plt.yticks(())
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)

plt.show()
