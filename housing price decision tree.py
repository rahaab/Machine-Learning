# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting house price from given id
id_no=2013
y_pred = regressor.predict([[id_no]])
print('House price for id', id_no,'=',y_pred)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Housing Price(Decision Tree Regression)')
plt.xlabel('ID')
plt.ylabel('Sale Price')
plt.show()
