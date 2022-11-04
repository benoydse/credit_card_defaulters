from os import listdir
import pandas
from application_logging.logger import AppLogger


class DataTransformPredict:
    """
    This class is used for transforming the Good Raw prediction Data before sending it to the Database.
    """

    def __init__(self):
        self.good_data_path = "Prediction_Raw_Files_Validated/Good_Raw"
        self.logger = AppLogger()

    def replace_missing_with_null(self):
        """
        This method replaces the missing values in columns with "NULL" to store in the table in the database.
        params: None
        Get all files from the good data path --> iterate over all files --> create a pandas df of each file -->
        wherever there are missing values it will be replaced with 'NULL' by pandas.
        returns: None
        """
        log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
        all_files = [f for f in listdir(self.good_data_path)]
        for file in all_files:
            data = pandas.read_csv(self.good_data_path + "/" + file)
            data.to_csv(self.good_data_path + "/" + file, index=None, header=True)
            self.logger.log(log_file, " %s: File Transformed successfully!!" % file)
        log_file.close()
