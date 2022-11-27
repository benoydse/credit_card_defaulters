import os
from pathlib import Path
import pandas as pd
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging.logger import AppLogger
from Prediction_Raw_Data_Validation.predictionDataValidation import PredictionDataValidation


class Prediction:
    """
    This class is used to perform prediction on the prediction data.
    """

    def __init__(self, path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = AppLogger()
        self.pred_data_val = PredictionDataValidation(path)
        if not os.path.exists('Prediction_Output_File'):
            os.makedirs(name='Prediction_Output_File')
        self._predicted_op_path = Path('Prediction_Output_File') / 'Predictions.csv'
        if os.path.exists(self._predicted_op_path):
            os.remove(self._predicted_op_path)

    def data_prediction(self):
        """
        Perform data prediction using the models.
        params: None
        Delete the existing prediction file if exists --> load the prediction data --> check for missing values in the
        dataset --> impute the missing values if present --> perform data scaling on the prediction data --> load the
        kmeans model --> predict clusters for the dataset --> create a column with cluster number --> find the correct
        model for the data --> perform prediction --> save to a csv file.
        returns: None
        """
        self.pred_data_val.delete_prediction_file()  # deletes the existing prediction file from last run
        self.log_writer.log(self.file_object, 'Start of Prediction')
        data_getter = data_loader_prediction.DataGetterPred(self.file_object, self.log_writer)
        data = data_getter.get_data()
        preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
        # check if missing values are present in the dataset
        is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
        # if missing values are there, replace them appropriately.
        if is_null_present:
            data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation
        x = preprocessor.scale_numerical_columns(data)
        file_loader = file_methods.FileOperation(self.file_object, self.log_writer)
        kmeans = file_loader.load_model('KMeans')
        clusters = kmeans.predict(x)
        x['clusters'] = clusters
        clusters = x['clusters'].unique()
        for i in clusters:
            cluster_data = x[x['clusters'] == i]
            cluster_data = cluster_data.drop(['clusters'], axis=1)
            model_name = file_loader.find_correct_model_file(i)
            model = file_loader.load_model(model_name)
            result = (model.predict(cluster_data))
            final = pd.DataFrame(list(zip(result)), columns=['Predictions'])
            final.to_csv(self._predicted_op_path, header=True, mode='a+')
            # appends result to prediction file
        self.log_writer.log(self.file_object, 'End of Prediction')
        return self._predicted_op_path
