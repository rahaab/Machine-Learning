# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

from sklearn.linear_model import LinearRegression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Global CO2 production')
plt.xlabel('Year')
plt.ylabel('Total CO2 production per year')
plt.show()

print ('2011=',lin_reg_2.predict(poly_reg.fit_transform([[2011]])))
print ('2012=',lin_reg_2.predict(poly_reg.fit_transform([[2012]])))
print ('2013=',lin_reg_2.predict(poly_reg.fit_transform([[2013]])))