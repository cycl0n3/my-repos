import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

sns.set_style("darkgrid")

data = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
print(data[0].shape)
print(data[1].shape)

# first 5 rows
print(data[0][:5])
print(data[1][:5])

# train test split
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=42)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# logistic regression
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
print(y_test)
print(y_pred)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# classification report
cr = classification_report(y_test, y_pred)
print(cr)

# calculate decision boundary
x1_min, x1_max = data[0][:, 0].min() - 1, data[0][:, 0].max() + 1
x2_min, x2_max = data[0][:, 1].min() - 1, data[0][:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))

# predict
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

# plot
plt.figure(figsize=(10, 8))
plt.contourf(xx1, xx2, Z, cmap=plt.cm.coolwarm, alpha=0.5)
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression')
plt.show()
