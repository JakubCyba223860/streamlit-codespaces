# %%
import pandas as pd
import xgboost as xgb
import  matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import seaborn as sns
from numpy import absolute
import numpy as np
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv("C:/Users/benjm/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/Datasets/finalized/features_missing_filledmean.csv")
crime = pd.read_csv("C:/Users/benjm/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/Datasets/finalized/label_crime_allmetrics.csv")
cross_join = pd.merge(df, crime, on=['NeighbourhoodCode', 'Year', 'Month'])

# %%
cross_join['NeighbourhoodCode'] = cross_join['NeighbourhoodCode'].astype(str)
cross_join['NeighbourhoodCode'] = LabelEncoder().fit_transform(cross_join['NeighbourhoodCode']) 

# %%
sns.heatmap(cross_join.isnull(), cbar=False, cmap='viridis')

# %%
selected = ['ResponseTimeScore',
       'SlowResponseTimePenalty', 'NuisanceReportsCriminal',
       'NuisanceReportsNonCriminal', 'lbm', 'afw', 'fys', 'onv', 'soc', 'vrz',
       'won', 'Migrated', 'NetLaborParticipation', 'FlexibleContracts',
       'SelfContract', 'PopulationEduLow', 'PopulationEduMedium',
       'PopulationEduHigh', 'y0-15%', 'y15-25%', 'y65-%',
       'Peoplewithmigrationbackgroud%', 'Averagepeopleperhousehold',
       'Populationdensitykm2', 'AverageWOZ-valueofhouses(x1000euro)',
       'Percentuninhabited(%)', 'Rentalproperies(%)', 'Tradeandcatering%',
       'Culture/recreationproperies%', 'Carsperhousehold', 'UrbanityLevel',
       'Inhabitants', 'Benches', 'Lights', 'POI', 'DayC', 'NightC',
       'Rain Days', 'Rainfall(mm)', 'Daylight hours']

X = cross_join[selected]

target = ['PropertyCrimesPerThousandInhabitants', 'ViolentCrimesPerThousandInhabitants',
          'BurglaryCrimesPerThousandInhabitants']

y = cross_join[target]

# %%
if False:
    # Instantiating the XGBoost regressor
    model = xgb.XGBRegressor(n_estimators=350, max_depth=60, learning_rate=0.3, subsample=1, colsample_bytree=1, random_state=1)

    cv = RepeatedKFold(n_splits=40, n_repeats=10, random_state=7)
    #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=2)

    mae_scores = []
    for column in target:
        # Get the index of the current target column
        target_index = target.index(column)
        
        # Extract the current target column
        y_current = y.iloc[:, target_index]
        
        # Calculate the MAE for the current target column
        scores = -1 * cross_val_score(model, X, y_current, scoring='neg_mean_absolute_error', cv=cv, n_jobs=14)
        mae_scores.append(scores)

    # Print the MAE for each target column
    for i, column in enumerate(target):
        print(f"MAE for {column}: {mae_scores[i].mean():.3f} ({mae_scores[i].std():.3f})")

# %%
model = xgb.XGBRegressor()
model.load_model('C:/Users/benjm/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team8/FinalDeliverable/app_data/artifacts/model_xgb.sav')

# %%
model.fit(X, y)

# Make predictions using the best model
predictions = model.predict(X)

# %%
predictions = predictions.flatten()
actual = y.values.flatten()

# %%
predictions = [max(0, pred) for pred in predictions]

# %%
results = pd.DataFrame({'Predicted': predictions, 'Actual': actual})

# %%
importances = model.feature_importances_

# Sort feature importances in descending order
sorted_indices = importances.argsort()[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = [selected[i] for i in sorted_indices]

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
plt.yticks(range(len(sorted_importances)), sorted_features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importances')
plt.show()

# %%
plt.plot(actual, label='Actual', linewidth=0.2)
plt.plot(predictions, label='Predicted', linewidth=0.2)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

differences = np.abs(predictions - actual)
std_deviation = np.std(differences)
num_bins = 100

plt.hist(differences, bins=num_bins, edgecolor='black', range=(0, .1))
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.title('Histogram of Standard Deviation')
plt.grid(True)
plt.show()

# %%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(actual, predictions)
print('Mean Squared Error (MSE):', mse)

r2 = r2_score(actual, predictions)
print('R-squared (R2):', r2)

mae = mean_absolute_error(actual, predictions)
print('Mean Absolute Error (MAE):', mae)

# %%
results.head(25)

# %%
#model.save_model('../FinalDeliverable/app_data/artifacts/model_xgb.sav')

# %%
X.min()


