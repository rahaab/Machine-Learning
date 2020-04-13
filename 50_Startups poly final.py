#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset['total spent']=dataset[['R&D Spend', 'Administration', 'Marketing Spend']].sum(axis=1)

#Separating the dataset for New York
XF = dataset.loc[(dataset.State=='Florida'),['total spent']].values
yF = dataset.loc[(dataset.State=='Florida'),['Profit']].values

# Separating the dataset for California
XC = dataset.loc[(dataset.State=='California'),['total spent']].values
yC = dataset.loc[(dataset.State=='California'),['Profit']].values

from sklearn.linear_model import LinearRegression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
XF_poly = poly_reg.fit_transform(XF)
poly_reg.fit(XF_poly,yF)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(XF_poly, yF)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
XC_poly = poly_reg.fit_transform(XC)
poly_reg.fit(XC_poly,yC)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(XC_poly, yC)

# Visualising the Polynomial Regression results
XFC_grid = np.arange(min(min(XF),min(XC)), max(max(XF),max(XC)), 0.1)
XFC_grid = XFC_grid.reshape((len(XFC_grid), 1))
plt.scatter(XF, yF, color = 'red')
plt.plot(XFC_grid, lin_reg_1.predict(poly_reg.fit_transform(XFC_grid)), color = 'red')
plt.scatter(XC, yC, color = 'blue')
plt.plot(XFC_grid, lin_reg_2.predict(poly_reg.fit_transform(XFC_grid)), color = 'blue')
plt.title('Profit vs Total Spenditure (red=Florida, blue=California)')
plt.xlabel('Total Spent')
plt.ylabel('Profit')
plt.show()
