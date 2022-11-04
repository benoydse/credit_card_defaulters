import pandas as pd


class DataGetter:
    """
    This class is used for obtaining the data from the database csv file for training.
    """

    def __init__(self, file_object, logger_object):
        self.training_file = 'Training_FileFromDB/InputFile.csv'
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        """
        This method reads the csv file of good data from the database and returns a pandas dataframe of the data.
        params: None
        Get the csv file and load it into a pandas dataframe --> return file.
        returns: data
        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the DataGetter class')
        data = pd.read_csv(self.training_file)  # reading the data file
        self.logger_object.log(self.file_object,
                               'Data Load Successful.Exited the get_data method of the DataGetter class')
        return data
