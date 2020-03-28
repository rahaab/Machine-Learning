from matplotlib import pyplot as plt
import pandas as pd

#importing data
data=pd.read_csv("dataset.csv")
x=data.iloc[:,2:3].values
y=data.iloc[:,3].values

#splitting data into test set and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=2/5)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(x_train, y_train)

# Predicting the Test set results
y_pred=regress.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'darkorange')
plt.plot(x_train, regress.predict(x_train), color = 'blue')
plt.title('Brain Weight vs head size (Training set)')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'darkorange')
plt.plot(x_train, regress.predict(x_train), color = 'black')
plt.title('Brain Weight against head size (Test set)')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.show()
