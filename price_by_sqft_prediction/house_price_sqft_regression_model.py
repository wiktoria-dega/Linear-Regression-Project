import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from MAPE_error import mean_absolute_percentage_error

client = MongoClient('mongodb://localhost:27017/')

db = client.test_price_sqft

collection = db.test

data = list(collection.find())

df_houseprice = pd.DataFrame(data)

df_houseprice.head()


X_train, X_test, y_train, y_test = train_test_split(
    df_houseprice.drop(columns=['price_by_sqft_log']),
    df_houseprice['price_by_sqft_log'],
    random_state=123,
    test_size=0.3)


scaler = MinMaxScaler()

X_train = X_train.drop(columns=['_id'])
X_test = X_test.drop(columns=['_id'])

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#linear regression model
model = LinearRegression()


model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)


print(f'MSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred))}')
print(f'RMSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred), squared=False)}')
print(f'R2-score: {r2_score(np.exp(y_test), np.exp(pred))}')
print(f'MAPE error for test data: {mean_absolute_percentage_error(np.exp(y_test), np.exp(pred))}')

plt.figure()
plt.scatter(list(range(len(y_test))), np.exp(y_test))
plt.scatter(list(range(len(pred))), np.exp(pred))
plt.legend(['real', 'pred'])


#decision tree model
tree_reg = DecisionTreeRegressor(max_depth=7)
tree_reg.fit(X_train_scaled, y_train)

pred_tree = tree_reg.predict(X_test_scaled)


print(f'MSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred_tree))}')
print(f'RMSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred_tree), squared = False)}')
print(f'R2-score: {r2_score(np.exp(y_test), np.exp(pred_tree))}')
print(f'MAPE error for test data: {mean_absolute_percentage_error(np.exp(y_test), np.exp(pred_tree))}')

plt.figure()
plt.scatter(list(range(len(y_test))), np.exp(y_test))
plt.scatter(list(range(len(pred))), np.exp(pred))
plt.legend(['real', 'pred_tree'])

# importance of features
for i,j in zip(tree_reg.feature_importances_, X_train.columns):
    print(i, j)
    
plt.figure()
plt.barh(X_train.columns, tree_reg.feature_importances_)


#random forest model
rnd_forest = RandomForestRegressor(n_estimators=200,
                                   max_depth=5,
                                   random_state=123)


rnd_forest.fit(X_train_scaled, y_train)

pred_forest = rnd_forest.predict(X_test_scaled)


print(f'MSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred_forest))}')
print(f'RMSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred_forest), squared = False)}')
print(f'R2-score: {r2_score(np.exp(y_test), np.exp(pred_forest))}')
print(f'MAPE error for test data: {mean_absolute_percentage_error(np.exp(y_test), np.exp(pred_forest))}')

# importance of features
for i,j in zip(rnd_forest.feature_importances_, X_train.columns):
    print(i, j)


#GridSearchCV
params = {
    'n_estimators': [5, 10, 25, 40, 50, 55, 60],
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7],
    'random_state': [123]
    }

rnd_forest_ = RandomForestRegressor()

gs = GridSearchCV(rnd_forest_, params, scoring='r2', verbose=2, cv=2)

gs.fit(X_train_scaled, y_train)

pred_best = gs.best_estimator_.predict(X_test_scaled)


print(f'MSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred_best))}')
print(f'RMSE error for test data: {mean_squared_error(np.exp(y_test), np.exp(pred_best), squared = False)}')
print(f'R2-score: {r2_score(np.exp(y_test), np.exp(pred_best))}')
print(f'MAPE error for test data: {mean_absolute_percentage_error(np.exp(y_test), np.exp(pred_best))}')

