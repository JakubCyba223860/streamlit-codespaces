import numpy as np
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from XGB import X,y,model

def measure_runtime(predict_func, data):
    start_time = time.time()
    _ = predict_func(data)  # Call the prediction function
    end_time = time.time()
    runtime = end_time - start_time
    return runtime

class TestModelPerformance:
    def setup_method(self, method):
        self.data = X
        self.target = y
        self.predictions = model.predict(X)

    def test_data_shape(self):
        # Ensure the data has the expected shape
        assert self.data.shape == (7232, 40)
        assert self.target.shape == (7232, 3)

    def test_X_columns(self):
        # Ensure the data has the expected columns
        expected_x_columns = ['ResponseTimeScore', 'SlowResponseTimePenalty', 'NuisanceReportsCriminal',
            'NuisanceReportsNonCriminal', 'lbm', 'afw', 'fys', 'onv', 'soc', 'vrz', 'won', 'Migrated',
            'NetLaborParticipation', 'FlexibleContracts', 'SelfContract', 'PopulationEduLow',
            'PopulationEduMedium', 'PopulationEduHigh', 'y0-15%', 'y15-25%', 'y65-%',
            'Peoplewithmigrationbackgroud%', 'Averagepeopleperhousehold', 'Populationdensitykm2',
            'AverageWOZ-valueofhouses(x1000euro)', 'Percentuninhabited(%)', 'Rentalproperies(%)',
            'Tradeandcatering%', 'Culture/recreationproperies%', 'Carsperhousehold', 'UrbanityLevel',
            'Inhabitants', 'Benches', 'Lights', 'POI', 'DayC', 'NightC', 'Rain Days', 'Rainfall(mm)',
            'Daylight hours']
        assert list(self.data.columns) == expected_x_columns

    def test_y_columns(self):
        # Ensure the data has the expected columns
        expected_y_columns = ['PropertyCrimesPerThousandInhabitants', 'ViolentCrimesPerThousandInhabitants',
            'BurglaryCrimesPerThousandInhabitants']
        assert list(self.target.columns) == expected_y_columns

    def test_data_duplicates(self):
        # Ensure there are no duplicate rows in the data
        duplicates = self.data.duplicated().sum()
        assert duplicates == 0

    def test_data_missing_values(self):
        # Ensure there are no missing values in the data
        missing_values = self.data.isnull().sum().sum()
        assert missing_values == 0

    def test_model_mean_absolute_error(self):
        # Calculate mean absolute error between predictions and actual target values
        mae = mean_absolute_error(self.target, self.predictions)
        np.testing.assert_almost_equal(mae, 0.001, decimal=2)

    def test_model_mean_squared_error(self):
        # Calculate mean squared error between predictions and actual target values
        mse = mean_squared_error(self.target, self.predictions)
        np.testing.assert_almost_equal(mse, 0.001, decimal=2)

    def test_model_r2_score(self):
        # Calculate R-squared score between predictions and target
        r2 = r2_score(self.target, self.predictions)
        assert r2 >= 0.9

    def test_model_maximum_error(self):
        # Calculate maximum absolute error between predictions and actual target values
        max_error = np.max(np.abs(self.target - self.predictions), axis=0)
        threshold = 2
        np.testing.assert_array_less(max_error, threshold)

    def test_model_explained_variance(self):
        # Calculate the percentage of explained variance by the model
        explained_variance = r2_score(self.target, self.predictions)
        assert explained_variance >= 0.9

    def test_model_prediction_consistency(self):
        # Re-run the model predictions and assert that the predictions remain the same
        new_predictions = model.predict(X)  # Replace model with your actual model
        np.testing.assert_array_equal(self.predictions, new_predictions)

    def test_model_runtime(self):
        # Measure the time taken by the model to make predictions on a sample dataset
        sample_data = self.data[:100]
        runtime = measure_runtime(model.predict, sample_data)  # Replace model with your actual model
        assert runtime < 5

logging.basicConfig(level=logging.INFO)
