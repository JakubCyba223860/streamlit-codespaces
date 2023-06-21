import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Read the CSV files
data = pd.read_csv('E:/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/Datasets/finalized/features_all_data.csv')
target = pd.read_csv('E:/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/Datasets/finalized/label_crime_allmetrics.csv')

target = target[['NeighbourhoodCode', 'Year', 'Month', 'PropertyCrimesTotal', 'PropertyCrimesRelPop', 'PropertyCrimesRelCity', 'ViolentCrimesTotal', 'ViolentCrimesRelPop', 'ViolentCrimesRelCity', 'BurglariesTotal', 'BurglariesRelPop', 'BurglariesRelCity']]

merged = target.merge(data, how='inner', on=['NeighbourhoodCode', 'Year', 'Month'])
merged = merged.drop(columns='NeighbourhoodCode')

# Convert the data to a numpy array
X = merged.drop(['PropertyCrimesTotal', 'PropertyCrimesRelPop', 'PropertyCrimesRelCity', 'ViolentCrimesTotal', 'ViolentCrimesRelPop', 'ViolentCrimesRelCity', 'BurglariesTotal', 'BurglariesRelPop', 'BurglariesRelCity'], axis=1)
y_property = merged['PropertyCrimesTotal']
y_property_relpop = merged['PropertyCrimesRelPop']
y_property_relcity = merged['PropertyCrimesRelCity']
y_violent = merged['ViolentCrimesTotal']
y_violent_relpop = merged['ViolentCrimesRelPop']
y_violent_relcity = merged['ViolentCrimesRelCity']
y_burglaries = merged['BurglariesTotal']
y_burglaries_relpop = merged['BurglariesRelPop']
y_burglaries_relcity = merged['BurglariesRelCity']

# Perform Z-score normalization on the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train, X_test, y_property_train, y_property_test, y_property_relpop_train, y_property_relpop_test, y_property_relcity_train, y_property_relcity_test, y_violent_train, y_violent_test, y_violent_relpop_train, y_violent_relpop_test, y_violent_relcity_train, y_violent_relcity_test, y_burglaries_train, y_burglaries_test, y_burglaries_relpop_train, y_burglaries_relpop_test, y_burglaries_relcity_train, y_burglaries_relcity_test = train_test_split(
    X_scaled, y_property, y_property_relpop, y_property_relcity, y_violent, y_violent_relpop, y_violent_relcity, y_burglaries, y_burglaries_relpop, y_burglaries_relcity, test_size=0.2, random_state=42)

# Create the regression models
models = [('gbr', GradientBoostingRegressor(random_state=42)),
          ('rfr', RandomForestRegressor(random_state=42)),
          ('xgb', XGBRegressor(random_state=42))]

# Create the VotingRegressor
voting_regressor = VotingRegressor(estimators=models)

# Define the parameter grid for GridSearchCV
param_grid = {'gbr__n_estimators': [100, 200],
              'gbr__learning_rate': [0.1, 0.05],
              'rfr__n_estimators': [100, 200],
              'xgb__n_estimators': [100, 200],
              'xgb__learning_rate': [0.1, 0.05]}

# Perform GridSearchCV
grid_search = GridSearchCV(voting_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_property_train)

# Get the best estimator and its predictions
best_estimator = grid_search.best_estimator_
y_property_pred = best_estimator.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_property_test, y_property_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_property_test, y_property_pred)
r2 = r2_score(y_property_test, y_property_pred)

# Print the evaluation metrics
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)
print('R^2 Score:', r2)

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_property_test.values, label='Actual')
plt.plot(y_property_pred, label='Predicted')
plt.xlabel('Samples')
plt.ylabel('Property Crimes')
plt.title('Actual vs. Predicted - Property Crimes')
plt.legend()
plt.show()

# Feature Importance
feature_importance = best_estimator.named_estimators_['xgb'].feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance - Property Crimes')
plt.show()
