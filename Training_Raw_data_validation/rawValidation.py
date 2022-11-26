from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from application_logging.logger import AppLogger


class RawDataValidation:
    """
    This class shall be used for handling all the validation done on the Raw Training Data.
    """

    def __init__(self, path):
        self.Batch_Directory = path
        self.schema_path = 'schema_training.json'
        self.logger = AppLogger()

    def values_from_schema(self):
        """
        This method is used to extract all relevant information from the pre-defined "Schema" file which is mostly given
        by clients.Schema also means rules to validate whether the data is in the correct format or not.
        params: None
        Open the schema.json file and store it into 'json_loaded_file' --> extract the values for rules from the
        dictionary --> save the values and return.
        returns: length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns
        """
        try:
            with open(self.schema_path, 'r') as f:
                json_loaded_file = json.load(f)
                f.close()
            length_of_date_stamp_in_file = json_loaded_file["LengthOfDateStampInFile"]
            length_of_time_stamp_in_file = json_loaded_file["LengthOfTimeStampInFile"]
            column_names = json_loaded_file['ColName']
            number_of_columns = json_loaded_file["NumberofColumns"]
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            message = "length_of_date_stamp_in_file:: %s" % length_of_date_stamp_in_file + "\t" + \
                      "length_of_time_stamp_in_file:: %s" % length_of_time_stamp_in_file + "\t " + \
                      "number_of_columns:: %s" % number_of_columns + "\n"
            self.logger.log(file, message)
            file.close()
        except ValueError:
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, "ValueError:Value not found inside schema_training.json")
            file.close()
            raise ValueError
        except KeyError:
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, "KeyError:Key value error incorrect key passed")
            file.close()
            raise KeyError
        except Exception as e:
            file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, str(e))
            file.close()
            raise e
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
        This method creates directories to store the Good Data and Bad Data files after validating the training data.
        params: None
        Use os.join() method with good_raw and bad_raw with Training_Raw_Files_validated to create a good_raw and
        bad_raw directory respectively.
        params: None
        returns: None
        """
        path = os.path.join("Training_Raw_files_validated/", "Good_Raw/")
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join("Training_Raw_files_validated/", "Bad_Raw/")
        if not os.path.isdir(path):
            os.makedirs(path)

    def delete_existing_good_data_training_folder(self):
        """
        This method is used to delete the good data directory made after storing the good data directory in the database
        This ensures space optimization.
        params: None
        Join the training raw files validated path with 'good_raw' path. This will give you the good raw training
        folder --> use shutils to delete the folder
        returns: None
        """
        path = 'Training_Raw_files_validated/'
        if os.path.isdir(path + 'Good_Raw/'):
            shutil.rmtree(path + 'Good_Raw/')
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "GoodRaw directory deleted successfully!!!")
            file.close()

    def delete_existing_bad_data_training_folder(self):
        """
        This method is used to delete the bad data directory made after storing the bad data directory in the database
        This ensures space optimization.
        params: None
        Join the training raw files validated path with 'bad_raw' path. This will give you the bad raw training
        folder --> use shutils to delete the folder
        returns: None
        """
        path = 'Training_Raw_files_validated/'
        if os.path.isdir(path + 'Bad_Raw/'):
            shutil.rmtree(path + 'Bad_Raw/')
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "BadRaw directory deleted before starting validation!!!")
            file.close()

    def move_bad_files_to_archive_bad(self):
        """
        This method deletes the directory made to store the Bad Data after moving the data in an archive folder.
        We archive the bad files to send them back to the client for any changes.
        params: None
        create a folder 'TrainingArchiveBadData' --> create a folder inside with 'BadData along with date and time -->
        copy all the files from bad data into this new folder.
        returns: None
        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        source = 'Training_Raw_files_validated/Bad_Raw/'
        if os.path.isdir(source):
            path = "TrainingArchiveBadData"
            if not os.path.isdir(path):
                os.makedirs(path)
            dest = 'TrainingArchiveBadData/BadData_' + str(date) + "_" + str(time)
            if not os.path.isdir(dest):
                os.makedirs(dest)
            files = os.listdir(source)
            for f in files:
                if f not in os.listdir(dest):
                    shutil.move(source + f, dest)
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Bad files moved to archive")
            path = 'Training_Raw_files_validated/'
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
        self.delete_existing_good_data_training_folder()
        # create new directories
        self.create_directory_for_good_bad_raw_data()
        all_files = [f for f in listdir(self.Batch_Directory)]
        f = open("Training_Logs/nameValidationLog.txt", 'a+')
        for filename in all_files:
            if re.match(regex, filename):  # ex file_name : creditCardFraud_28011960_120210.csv
                name_and_csv = filename.split('.')  # ['creditCardFraud_28011960_120210', 'csv']
                name_dstamp_tstamp = name_and_csv[0].split('_')  # ['creditCardFraud', '28011960', '120210']
                if len(name_dstamp_tstamp[1]) == length_of_date_stamp_in_file:
                    if len(name_dstamp_tstamp[2]) == length_of_time_stamp_in_file:
                        shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
                        self.logger.log(f, "Valid File name!! File moved to GoodRaw Folder :: %s" % filename)
                    else:
                        shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
            else:
                shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
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
        f = open("Training_Logs/columnValidationLog.txt", 'a+')
        self.logger.log(f, "Column Length Validation Started!!")
        for file in listdir('Training_Raw_files_validated/Good_Raw/'):
            df = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
            if df.shape[1] == number_of_columns:
                pass
            else:
                shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
        self.logger.log(f, "Column Length Validation Completed!!")
        f.close()

    def validate_if_entire_column_has_missing_values(self):
        """
        Checks whether all the values in the column are null values.
        params: None
        Iterate over files in 'Training_Raw_files_validated/Good_Raw' --> create a df of the file --> iterate over
        columns of the dataframe --> if the number of rows in a column - the sum of non-null rows = number of rows -->
        then the entire row contains null values --> move to bad folder.
        returns: None
        """
        f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
        self.logger.log(f, "Missing Values Validation Started!!")
        for file in listdir('Training_Raw_files_validated/Good_Raw/'):
            df = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
            count = 0
            for column in df:
                # df[column].count() returns the count of non-Null rows
                if (len(df[column]) - df[column].count()) == len(df[column]):
                    count += 1
                    shutil.move("Training_Raw_files_validated/Good_Raw/" + file,
                                "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid Column for the file!! File moved to Bad Raw Folder :: %s" % file)
                    break
            if count == 0:
                # df.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                df.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
        f.close()
