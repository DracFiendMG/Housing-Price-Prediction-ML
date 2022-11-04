from math import sqrt

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# read file
data = pd.read_csv("housing.csv")
# print few rows of this data
print(data.head())

# print(data.isnull().sum())

data.total_bedrooms = data.total_bedrooms.fillna(data.total_bedrooms.mean())
# print(data.isnull().sum())

le = LabelEncoder()
data['ocean_proximity']=le.fit_transform(data['ocean_proximity'])


columns = data.columns

scaler = StandardScaler()

scaler_data = scaler.fit_transform(data)
scaler_data = pd.DataFrame(data, columns=columns)
print(scaler_data.head())

X_Features=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']
X=scaler_data[X_Features]
Y=scaler_data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Linear Regression
LR = LinearRegression()
LR.fit(x_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
y_predict = LR.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))

# Decision Tree
DTR=DecisionTreeRegressor()
DTR.fit(x_train, y_train)

y_predict = DTR.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))

# Random Forest Regression
RF=RandomForestRegressor()
RF.fit(x_train, y_train)
y_predict = RF.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))

x_train_Income=x_train[['median_income']]
x_test_Income=x_test[['median_income']]

print(x_train_Income.shape)
print(y_train.shape)

LR=LinearRegression()
LR.fit(x_train_Income, y_train)
y_predict = LR.predict(x_test_Income)

print(LR.intercept_, LR.coef_)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))

scaler_data.plot(kind='scatter',x='median_income',y='median_house_value')
plt.plot(x_test_Income,y_predict,c='red',linewidth=2)

plt.show()
