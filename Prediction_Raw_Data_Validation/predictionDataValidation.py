from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from application_logging.logger import AppLogger


class PredictionDataValidation:
    """
    This class is used for handling all the validation done on the Raw Prediction Data!!.
    """

    def __init__(self, path):
        self.Batch_Directory = path
        self.schema_path = 'schema_prediction.json'
        self.logger = AppLogger()

    def values_from_schema(self):
        """
        This method is used to extract all relevant information from the pre-defined "Schema" file which is mostly given
        by clients. Schema also means rules to validate whether the data is in the correct format or not.
        params: None
        Open the schema.json file and store it into 'json_loaded_file' --> extract the values for rules from the
        dictionary --> save the values and return.
        returns: length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns
        """
        with open(self.schema_path, 'r') as f:
            dic = json.load(f)
            f.close()
        length_of_date_stamp_in_file = dic['length_of_date_stamp_in_file']
        length_of_time_stamp_in_file = dic['length_of_time_stamp_in_file']
        column_names = dic['ColName']
        number_of_columns = dic['number_of_columns']
        file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
        message = "length_of_date_stamp_in_file:: %s" % length_of_date_stamp_in_file + "\t" + \
                  "length_of_time_stamp_in_file:: %s" % length_of_time_stamp_in_file + "\t " + \
                  "number_of_columns:: %s" % number_of_columns + "\n"
        self.logger.log(file, message)
        file.close()
        return length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns

    @staticmethod
    def manual_regex_creation():
        """
        This method contains a manually defined regular expression based on the "FileName" given in "Schema.json" file
        which is later used to validate the filename of the training data.
        params: None
        returns: Regex pattern
        """
        regex = "['creditCardFraud']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    @staticmethod
    def create_directory_for_good_bad_raw_data():
        """
        This method creates directories to store the Good Data and Bad Data files after validating the prediction data.
        params: None
        Use os.join() method with good_raw and bad_raw with prediction_Raw_Files_validated to create a good_raw and
        bad_raw directory respectively.
        returns: None
        """
        path = os.path.join("Prediction_Raw_Files_Validated/", "Good_Raw/")
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join("Prediction_Raw_Files_Validated/", "Bad_Raw/")
        if not os.path.isdir(path):
            os.makedirs(path)

    def delete_existing_good_data_prediction_folder(self):
        """
        This method is used to delete the good data directory made after storing the good data directory in the database
        This ensures space optimization.
        params: None
        Join the prediction raw files validated path with 'good_raw' path. This will give you the good raw prediction
        folder --> use shutils to delete the folder
        returns: None
        """
        path = 'Prediction_Raw_Files_Validated/'
        if os.path.isdir(path + 'Good_Raw/'):
            shutil.rmtree(path + 'Good_Raw/')
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "GoodRaw directory deleted successfully!!!")
            file.close()

    def delete_existing_bad_data_training_folder(self):
        """
        This method is used to delete the bad data directory made after storing the bad data directory in the database
        This ensures space optimization.
        params: None
        Join the prediction raw files validated path with 'bad_raw' path. This will give you the bad raw training
        folder --> use shutils to delete the folder
        returns: None
        """
        path = 'Prediction_Raw_Files_Validated/'
        if os.path.isdir(path + 'Bad_Raw/'):
            shutil.rmtree(path + 'Bad_Raw/')
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "BadRaw directory deleted before starting validation!!!")
            file.close()

    def move_bad_files_to_archive_bad(self):
        """
        This method deletes the directory made to store the Bad Data after moving the data in an archive folder.
        We archive the bad files to send them back to the client for any changes.
        params: None
        create a folder 'PredictionArchiveBadData' --> create a folder inside with 'BadData along with date and time -->
        copy all the files from bad data into this new folder.
        returns: None
        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        path = "PredictionArchivedBadData"
        if not os.path.isdir(path):
            os.makedirs(path)
        source = 'Prediction_Raw_Files_Validated/Bad_Raw/'
        dest = 'PredictionArchivedBadData/BadData_' + str(date) + "_" + str(time)
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(source)
        for f in files:
            if f not in os.listdir(dest):
                shutil.move(source + f, dest)
        file = open("Prediction_Logs/GeneralLog.txt", 'a+')
        self.logger.log(file, "Bad files moved to archive")
        path = 'Prediction_Raw_Files_Validated/'
        if os.path.isdir(path + 'Bad_Raw/'):
            shutil.rmtree(path + 'Bad_Raw/')
        self.logger.log(file, "Bad Raw Data Folder Deleted successfully!!")
        file.close()

    def validation_file_name_raw(self, regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file):
        """
        This function validates the name of the training csv files as per given name according to the schema.
        params: regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file
        Delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted -->
        create folders for good and bad raw data --> get all the files in the path --> iterate over files --> compare
        file name with the regex pattern that we created earlier --> if it matches the regex then split the filename
        at the .csv --> next split the file name without the csv part on the underscore in file name --> check if the
        length of the date and time stamp match that of the regex --> if it fulfills both conditions copy the files to
        "Training_Raw_files_validated/Good_Raw" --> otherwise copy to "Training_Raw_files_validated/Bad_Raw"
        returns: None
        """
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.delete_existing_bad_data_training_folder()
        self.delete_existing_good_data_prediction_folder()
        self.create_directory_for_good_bad_raw_data()
        all_files = [f for f in listdir(self.Batch_Directory)]
        f = open("Prediction_Logs/nameValidationLog.txt", 'a+')
        for filename in all_files:
            if re.match(regex, filename):
                name_csv = re.split('.csv', filename)
                name_dstamp_tstamp = (re.split('_', name_csv[0]))
                if len(name_dstamp_tstamp[1]) == length_of_date_stamp_in_file:
                    if len(name_dstamp_tstamp[2]) == length_of_time_stamp_in_file:
                        shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Good_Raw")
                        self.logger.log(f, "Valid File name!! File moved to GoodRaw Folder :: %s" % filename)
                    else:
                        shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
            else:
                shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
        f.close()

    def validate_number_of_columns_in_file(self, number_of_columns):
        """
        This method is used to check whether the file has the correct number of columns.
        params: Number of columns
        Iterate over files in 'Training_Raw_files_validated/Good_Raw/' --> create a pandas dataframe of the file -->
        check the shape of the dataframe --> shape rows 'rows, columns' so take the second element and compare it with
        number of columns --> if false move the file into "Training_Raw_files_validated/Bad_Raw".
        returns: None
        """
        f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
        self.logger.log(f, "Column Length Validation Started!!")
        for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
            csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
            if csv.shape[1] == number_of_columns:
                csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
            else:
                shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                            "Prediction_Raw_Files_Validated/Bad_Raw")
                self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
        self.logger.log(f, "Column Length Validation Completed!!")
        f.close()

    @staticmethod
    def delete_prediction_file():

        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')

    def validate_if_entire_column_has_missing_values(self):
        """
        Checks whether all the values in the column are null values.
        params: None
        Iterate over files in 'Prediction_Raw_files_validated/Good_Raw' --> create a df of the file --> iterate over
        columns of the dataframe --> if the number of rows in a column - the sum of non-null rows = number of rows -->
        then the entire row contains null values --> move to bad folder.
        returns: None
        """
        f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
        self.logger.log(f, "Missing Values Validation Started!!")
        for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
            csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
            count = 0
            for columns in csv:
                if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                    count += 1
                    shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                                "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.logger.log(f,
                                    "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                    break
            if count == 0:
                csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
        f.close()
