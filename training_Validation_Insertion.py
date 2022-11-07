from Training_Raw_data_validation.rawValidation import RawDataValidation
from DataTypeValidation_Insertion_Training.DataTypeValidation import DBOperation
from DataTransform_Training.DataTransformation import DataTransform
from application_logging import logger
import os


class TrainValidation:
    """
    This class has been written for the validation of the training data given by the client.
    """

    def __init__(self, path):
        self.raw_data = RawDataValidation(path)
        self.data_transform = DataTransform()
        self.dBOperation = DBOperation()
        self.cwd = os.getcwd()
        self.file_object = open(self.cwd + 'Training_Main_Log.txt', 'a+')
        self.log_writer = logger.AppLogger()

    def train_validation(self):
        """
        Perform data validation on the training data.
        params: None
        Get the data specifications from the training schema --> perform data validation.
        returns: None
        """
        self.log_writer.log(self.file_object, 'Start of Validation on files for Training')
        length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns = \
            self.raw_data.values_from_schema()
        regex = self.raw_data.manual_regex_creation()
        self.raw_data.validation_file_name_raw(regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file)
        self.raw_data.validate_number_of_columns_in_file(number_of_columns)
        self.raw_data.validate_if_entire_column_has_missing_values()
        self.data_transform.replace_missing_with_null()
        self.dBOperation.create_table_in_db('Training', column_names)
        self.dBOperation.insert_good_data_into_table('Training')
        self.raw_data.delete_existing_good_data_training_folder()
        self.raw_data.move_bad_files_to_archive_bad()
        self.dBOperation.export_good_data_to_csv('Training')
        self.file_object.close()
