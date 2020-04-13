# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset for GCAG
dataset = pd.read_csv('annual_temp.csv')
X1 = dataset.loc[(dataset.Source=='GCAG'),['Year']].values
y1 = dataset.loc[(dataset.Source=='GCAG'),['Mean']].values

from sklearn.linear_model import LinearRegression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X1)
poly_reg.fit(X_poly,y1)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_poly, y1)

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X1), max(X1))
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X1, y1, color = 'red')
plt.plot(X_grid, lin_reg_1.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Annual temp of GCAG')
plt.xlabel('Year')
plt.ylabel('Mean temp')
plt.show()

#predicting temp of 2016 and 2017 for GCAG
print ('GCAG 2016=',lin_reg_1.predict(poly_reg.fit_transform([[2016]])))
print ('GCAG 2017=',lin_reg_1.predict(poly_reg.fit_transform([[2017]])))

# Importing the dataset for GISTEMP
X2 = dataset.loc[(dataset.Source=='GISTEMP'),['Year']].values
y2 = dataset.loc[(dataset.Source=='GISTEMP'),['Mean']].values

# Fitting Polynomial Regression to the dataset
poly_reg2 = PolynomialFeatures(degree = 4)
X_poly2 = poly_reg2.fit_transform(X2)
poly_reg2.fit(X_poly2,y2)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly2, y2)

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X2), max(X2))
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X2, y2, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg2.fit_transform(X_grid)), color = 'blue')
plt.title('Annual temp of GISTEMP')
plt.xlabel('Year')
plt.ylabel('Mean temp')
plt.show()

#predicting temp of 2016 and 2017 for GISTEMP
print ('GISTEMP 2016=',lin_reg_2.predict(poly_reg2.fit_transform([[2016]])))
print ('GISTEMP 2017=',lin_reg_2.predict(poly_reg2.fit_transform([[2017]])))     
