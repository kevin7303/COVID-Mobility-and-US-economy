import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv('data_clean.csv')
df.columns

#One Hot Encoding
df = pd.get_dummies(df, columns=['sub_region_1'], drop_first=True)
df.columns

#Split into X and y
X = df.drop(['Close', 'date', 'month'], axis = 1)
y = df['Close']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
X.shape[0] == y.shape[0]

#Models

#Ridge Regression
el = ElasticNet()
el.fit(X_train, y_train)

y_pred = el.predict(X_test)
rmse = mean_squared_error(y_pred, y_test, squared = False)
rmse

#Random Forest
rf = RandomForestRegressor(verbose = 1, n_jobs = -1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_pred,y_test, squared = False)
rmse

#XGBoost
gbm = GradientBoostingRegressor(verbose = 1)
gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)
rmse = mean_squared_error(y_pred, y_test, squared = False)
rmse

#Hyper Parameter Tuning
#Random Forest
rf = RandomForestRegressor()
rf_grid = {"n_estimators": [20, 50], "max_depth": [5, 10, 20], 'min_samples_split': [2, 5, 10]}
#Xgboost Classifier
gbm = GradientBoostingRegressor()
gbm_grid = {'learning_rate': [ 0.03, 0.05], 'max_depth': [5, 10, 20], 'n_estimators': [20, 50]}

rf_search = RandomizedSearchCV(rf, rf_grid, random_state= 99, scoring = 'neg_mean_squared_error', n_iter = 10, n_jobs = -1, verbose = 1)
rf_search.fit(X, y)

rf_search.best_params_


gbm_search = RandomizedSearchCV(gbm, gbm_grid, random_state= 99, scoring = 'neg_mean_squared_error', n_iter = 10, n_jobs = -1, verbose = 1)
gbm_search.fit(X, y)
gbm_search.best_params_

#Random Forest
rf = RandomForestRegressor(verbose=1, n_jobs=-1, n_estimators = 200, min_samples_split= 2, max_depth= 10)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_pred, y_test, squared=False)
rmse

#GradientBoostingRegression <- Best Result
gbm = GradientBoostingRegressor(verbose=1, n_estimators = 200, max_depth = 10, learning_rate = 0.05)
gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)
rmse = mean_squared_error(y_pred, y_test, squared=False)
rmse


#Graphing Feature Importance
col_name = ['retail_and_recreation_change', 'grocery_and_pharmacy_change', 'parks_change', 'transit_stations_change', 'workplaces_change', 'residential_change', 'sub_region_1_Alaska', 'sub_region_1_Arizona', 'sub_region_1_Arkansas', 'sub_region_1_California', 'sub_region_1_Colorado', 'sub_region_1_Connecticut', 'sub_region_1_Delaware', 'sub_region_1_District of Columbia', 'sub_region_1_Florida', 'sub_region_1_Georgia', 'sub_region_1_Hawaii', 'sub_region_1_Idaho', 'sub_region_1_Illinois', 'sub_region_1_Indiana', 'sub_region_1_Iowa', 'sub_region_1_Kansas', 'sub_region_1_Kentucky', 'sub_region_1_Louisiana', 'sub_region_1_Maine', 'sub_region_1_Maryland', 'sub_region_1_Massachusetts', 'sub_region_1_Michigan', 'sub_region_1_Minnesota', 'sub_region_1_Mississippi', 'sub_region_1_Missouri', 'sub_region_1_Montana', 'sub_region_1_Nebraska', 'sub_region_1_Nevada', 'sub_region_1_New Hampshire', 'sub_region_1_New Jersey', 'sub_region_1_New Mexico', 'sub_region_1_New York', 'sub_region_1_North Carolina', 'sub_region_1_North Dakota', 'sub_region_1_Ohio', 'sub_region_1_Oklahoma', 'sub_region_1_Oregon', 'sub_region_1_Pennsylvania', 'sub_region_1_Rhode Island', 'sub_region_1_South Carolina', 'sub_region_1_South Dakota', 'sub_region_1_Tennessee', 'sub_region_1_Texas', 'sub_region_1_Utah', 'sub_region_1_Vermont', 'sub_region_1_Virginia', 'sub_region_1_Washington', 'sub_region_1_West Virginia', 'sub_region_1_Wisconsin', 'sub_region_1_Wyoming']

plt.figure(figsize=(30, 8))
pd.Series(gbm.feature_importances_, index=col_name).nlargest(6).plot(kind='barh')

plt.savefig('feature_importance.png')
