import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from application_logging.logger import AppLogger


class DBOperation:
    """This class is used for handling all the SQL operations."""

    def __init__(self):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.logger = AppLogger()

    def create_data_base_connection(self, database_name):
        """
        Description: This method creates the database with the given name and if Database already exists then opens the
        connection to the DB.
        params: database_name
        Create database --> open the connection.
        returns: None
        """
        conn = sqlite3.connect(self.path + database_name + '.db')
        file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
        self.logger.log(file, "Opened %s database successfully" % database_name)
        file.close()
        return conn

    def create_table_in_db(self, database_name, column_names):
        """
        Creates a table in the database.
        params: database_name, column_names
        creates a database connection -->
        returns: None
        """
        conn = self.create_data_base_connection(database_name)
        c = conn.cursor()
        c.execute("SELECT count(name)  FROM sqlite_master WHERE type = 'table' AND name = 'Good_Raw_Data'")
        if c.fetchone()[0] == 1:
            conn.close()
            file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Tables created successfully!!")
            file.close()
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully" % database_name)
            file.close()
        else:
            for key in column_names.keys():
                _type = column_names[key]
                # in try block we check if the table exists, if yes then add columns to the table
                # else in the except block we will create the table
                try:
                    conn.execute(
                        'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,
                                                                                                 dataType=_type))
                except:
                    conn.execute('CREATE TABLE  Good_Raw_Data ({column_name} {dataType})'.format(column_name=key,
                                                                                                 dataType=_type))
            conn.close()
            file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Tables created successfully!!")
            file.close()
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully" % database_name)
            file.close()

    def insert_good_data_into_table(self, database):
        """
        This method inserts the Good data files from the Good_Raw folder into the above created table in the database.
        params: database
        Iterate over all files in the good data directory --> open each file --> get the values in each line -->
        write to the database table --> close the connection.
        returns: None
        """
        conn = self.create_data_base_connection(database)
        good_file_path = self.goodFilePath
        bad_file_path = self.badFilePath
        all_files = [f for f in listdir(good_file_path)]
        log_file = open("Training_Logs/DbInsertLog.txt", 'a+')
        for file in all_files:
            try:
                with open(good_file_path + '/' + file, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for idx, line in enumerate(reader):
                        for values in line:
                            conn.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=values))
                            self.logger.log(log_file, " %s: File loaded successfully!!" % file)
                            conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.log(log_file, "Error while creating table: %s " % e)
                shutil.move(good_file_path + '/' + file, bad_file_path)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()
                conn.close()
        conn.close()
        log_file.close()

    def export_good_data_to_csv(self, database):
        """
        This method exports the data in the good_data table to a csv file.
        params: database
        Select the data from Good_Raw_Data directory --> get the headers of the file --> create the csv file
        --> write headers to csv --> write the data to the csv file.
        returns: None
        """
        file_from_db = 'Training_FileFromDB/'
        file_name = 'InputFile.csv'
        log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
        conn = self.create_data_base_connection(database)
        sql_select = "SELECT *  FROM Good_Raw_Data"
        cursor = conn.cursor()
        cursor.execute(sql_select)
        results = cursor.fetchall()
        # Get the headers of the csv file
        headers = [i[0] for i in cursor.description]
        # Make the CSV output directory
        if not os.path.isdir(file_from_db):
            os.makedirs(file_from_db)
        # Open CSV file for writing.
        csv_file = csv.writer(open(file_from_db + file_name, 'w', newline=''), delimiter=',',
                              lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')
        # Add the headers and data to the CSV file.
        csv_file.writerow(headers)
        csv_file.writerows(results)
        self.logger.log(log_file, "File exported successfully!!!")
        log_file.close()
