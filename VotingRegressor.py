# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# %%

# Read the CSV file
data = pd.read_csv('E:/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/Datasets/finalized/features_all_data.csv')
target = pd.read_csv('E:/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/Datasets/finalized/label_crime_total_perinhabitant.csv')


# %%
print(target.columns)

# %%
print(data.columns)

# %%
target

# %%
data


# %%
merged = target.merge(data, how='inner', on=['NeighbourhoodCode','Year','Month'])

# %%
merged

# %%
merged = merged.drop(columns='NeighbourhoodCode')

# %%

# Convert the data to a numpy array
X = merged.drop('CrimeRate',axis=1)
y = merged['CrimeRate']


# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# %%
regressor1 = RandomForestRegressor()
regressor2 = DecisionTreeRegressor()
regressor3 = KNeighborsRegressor()

voting_regressor = VotingRegressor([('rf', regressor1), ('dt', regressor2), ('knn', regressor3)])

voting_regressor.fit(X_train, y_train)

predictions = voting_regressor.predict(X_test)

# %%
predictions

# %%
from sklearn.metrics import mean_squared_error, r2_score

# Assuming y_test contains the true target values
predictions = voting_regressor.predict(X_test)

# Calculate R-squared score for each model
model_scores = {}
for name, model in voting_regressor.named_estimators_.items():
    model_predictions = model.predict(X_test)
    model_scores[name] = r2_score(y_test, model_predictions)

# Print the R-squared score for each model
for name, score in model_scores.items():
    print(f"{name} R-squared score: {score}")


# %%
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# %%
mse

# %%
r2

# %%
# Define the parameter grid for each individual model
param_grid_1 = {
    'random_state':[0],

    'max_depth':[3, 5, 7, 10],

    'n_estimators': [27, 50, 75, 100, 200, 300],

    'max_leaf_nodes':[20,40,50]
}

param_grid_2 = {
    'max_depth': [None, 5, 10, 7]
}

param_grid_3 = {
    'n_neighbors': [3, 5, 7]
}

# Perform grid search on each individual model
grid_search_1 = GridSearchCV(estimator=regressor1, param_grid=param_grid_1, cv=5)
grid_search_1.fit(X_train, y_train)

grid_search_2 = GridSearchCV(estimator=regressor2, param_grid=param_grid_2, cv=5)
grid_search_2.fit(X_train, y_train)

grid_search_3 = GridSearchCV(estimator=regressor3, param_grid=param_grid_3, cv=5)
grid_search_3.fit(X_train, y_train)

# Create the voting regressor with the optimized models
voting_regressor = VotingRegressor([
    ('rf', grid_search_1.best_estimator_),
    ('dt', grid_search_2.best_estimator_),
    ('knn', grid_search_3.best_estimator_)
])

# Fit the voting regressor on the training data
voting_regressor.fit(X_train, y_train)

# Make predictions using the voting regressor
predictions = voting_regressor.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error, r2_score

# Assuming y_test contains the true target values
predictions = voting_regressor.predict(X_test)

# Calculate R-squared score for each model
model_scores = {}
for name, model in voting_regressor.named_estimators_.items():
    model_predictions = model.predict(X_test)
    model_scores[name] = r2_score(y_test, model_predictions)

# Print the R-squared score for each model
for name, score in model_scores.items():
    print(f"{name} R-squared score: {score}")


# %%
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# %%
mse

# %%
r2

# %%
predictions


voting_regressor = VotingRegressor([('rf', regressor1), ('dt', regressor2), ('knn', regressor3)])
voting_regressor.fit(X_train, y_train)

# Get feature importances from RandomForestRegressor
importances = voting_regressor.named_estimators_['rf'].feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color='b', align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(y_test_sorted, predictions_sorted)
print('Mean Squared Error (MSE):', mse)

r2 = r2_score(y_test_sorted, predictions_sorted)
print('R-squared (R2):', r2)

mae = mean_absolute_error(y_test_sorted, predictions_sorted)
print('Mean Absolute Error (MAE):', mae)