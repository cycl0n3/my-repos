import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

from sklearn.linear_model import (
    LinearRegression, 
    LogisticRegression
)

from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Regression
regr1 = GradientBoostingRegressor()
regr2 = RandomForestRegressor()
regr3 = LinearRegression()

regr1.fit(X_train, y_train)
regr2.fit(X_train, y_train)
regr3.fit(X_train, y_train)

# Voting
ereg = VotingRegressor(estimators=[("gb", regr1), ("rf", regr2), ("lr", regr3)])
ereg.fit(X_train, y_train)

# Predict
y_pred1 = regr1.predict(X_test)
y_pred2 = regr2.predict(X_test)
y_pred3 = regr3.predict(X_test)
y_pred4 = ereg.predict(X_test)

# Plot
plt.figure(figsize=(12, 8))

plt.plot(y_pred1, "gd", label="GradientBoostingClassifier")
plt.plot(y_pred2, "b^", label="RandomForestRegressor")
plt.plot(y_pred3, "ys", label="LinearRegression")
plt.plot(y_pred4, "r*", label="VotingRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("Predicted")
plt.xlabel("Training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")

plt.show()
