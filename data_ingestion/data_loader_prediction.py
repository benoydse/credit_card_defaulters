import pandas as pd


class DataGetterPred:
    """
    This class is used for obtaining the data from the source for prediction.
    """

    def __init__(self, file_object, logger_object):
        self.prediction_file = 'Prediction_FileFromDB/InputFile.csv'
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        """
        This method reads the csv file of prediction data from the database and returns a pandas dataframe of the data.
        params: None
        Get the csv file and load it into a pandas dataframe --> return file.
        returns: data
        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the DataGetter class')
        data = pd.read_csv(self.prediction_file)  # reading the data file
        self.logger_object.log(self.file_object,
                               'Data Load Successful.Exited the get_data method of the DataGetter class')
        return data
