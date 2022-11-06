import shutil
import sqlite3
from os import listdir
import os
import csv
from application_logging.logger import AppLogger


class DBOperation:
    """
    This class is used for handling all the SQL operations for the prediction data.
    """

    def __init__(self):
        self.path = 'Prediction_Database/'
        self.bad_file_path = "Prediction_Raw_Files_Validated/Bad_Raw"
        self.good_file_path = "Prediction_Raw_Files_Validated/Good_Raw"
        self.logger = AppLogger()

    def create_database_connection(self, database_name):
        """
        Description: This method creates the database with the given name and if Database already exists then opens the
        connection to the DB.
        params: database_name
        Create database --> open the connection.
        returns: None
        """
        conn = sqlite3.connect(self.path + database_name + '.db')
        file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
        self.logger.log(file, "Opened %s database successfully" % database_name)
        file.close()
        return conn

    def create_table_in_db(self, database_name, column_names):
        """
        This method creates a table in the given database which will be used to insert the Good data after raw
        data validation.
        params: database_name, columns_names
        create a connection to the db --> drop the table if it exists --> iterate over the column names keys -->
        get the datatype --> if the table exists --> add columns to it --> else create the table --> close the
        connection.
        returns: None
        """
        conn = self.create_database_connection(database_name)
        conn.execute('DROP TABLE IF EXISTS Good_Raw_Data;')
        for key in column_names.keys():
            data_type = column_names[key]
            # in try block we check if the table exists, if yes then add columns to the table
            # else in catch block we create the table
            try:
                # cur = cur.execute("SELECT name FROM {dbName} WHERE data_type='table' AND name='Good_Raw_Data'
                # ".format(dbName=database_name))
                conn.execute(
                    'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,
                                                                                             dataType=data_type))
            except:
                conn.execute(
                    'CREATE TABLE  Good_Raw_Data ({column_name} {dataType})'.format(column_name=key,
                                                                                    dataType=data_type))
        conn.close()
        file = open("Prediction_Logs/DbTableCreateLog.txt", 'a+')
        self.logger.log(file, "Tables created successfully!!")
        file.close()
        file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
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
        conn = self.create_database_connection(database)
        all_files = [f for f in listdir(self.good_file_path)]
        log_file = open("Prediction_Logs/DbInsertLog.txt", 'a+')
        for file in all_files:
            try:
                with open(self.good_file_path + '/' + file, "r") as f:
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
                shutil.move(self.good_file_path + '/' + file, self.bad_file_path)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()
                conn.close()
                raise e
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
        file_from_db = 'Prediction_FileFromDB/'
        file_name = 'InputFile.csv'
        log_file = open("Prediction_Logs/ExportToCsv.txt", 'a+')
        conn = self.create_database_connection(database)
        sql_select = "SELECT *  FROM Good_Raw_Data"
        cursor = conn.cursor()
        cursor.execute(sql_select)
        results = cursor.fetchall()
        # Get the headers of the csv file
        headers = [i[0] for i in cursor.description]
        # Make the CSV ouput directory
        if not os.path.isdir(file_from_db):
            os.makedirs(file_from_db)
        # Open CSV file for writing.
        csv_file = csv.writer(open(file_from_db + file_name, 'w', newline=''), delimiter=',',
                              lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')
        # Add the headers and data to the CSV file.
        csv_file.writerow(headers)
        csv_file.writerows(results)
        self.logger.log(log_file, "File exported successfully!!!")
