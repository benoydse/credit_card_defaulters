from datetime import datetime
from Prediction_Raw_Data_Validation.predictionDataValidation import PredictionDataValidation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import DBOperation
from DataTransformation_Prediction.DataTransformationPrediction import DataTransformPredict
from application_logging import logger


class PredictionValidation:

    def __init__(self, path):
        self.raw_data = PredictionDataValidation(path)
        self.dataTransform = DataTransformPredict()
        self.dBOperation = DBOperation()
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.AppLogger()

    def prediction_validation(self):
        """
        Perform data validation on the prediction data.
        params: None
        Get the specifications from schema --> perform all validation steps
        returns: None
        """
        self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')
        length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns = \
            self.raw_data.values_from_schema()
        regex = self.raw_data.manual_regex_creation()
        self.raw_data.validation_file_name_raw(regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file)
        self.raw_data.validate_number_of_columns_in_file(number_of_columns)
        self.raw_data.validate_if_entire_column_has_missing_values()
        self.dataTransform.replace_missing_with_null()
        self.dBOperation.create_table_in_db('Prediction', column_names)
        self.dBOperation.insert_good_data_into_table('Prediction')
        self.raw_data.delete_existing_good_data_prediction_folder()
        self.raw_data.move_bad_files_to_archive_bad()
        self.dBOperation.export_good_data_to_csv('Prediction')
        self.log_writer.log(self.file_object, 'End of validation on files for prediction.')
