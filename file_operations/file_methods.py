import pickle
import os
import shutil


class FileOperation:
    """
    This class shall be used to save the model after training and load the saved model for prediction.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory = 'models/'

    def save_model(self, model, filename):
        """
        Save the model files into the 'models' directory.
        params: model, filename
        Create a file with required file name inside the model directory --> if path exists --> delete it --> else
        create the path --> save the model.
        returns: status
        """
        self.logger_object.log(self.file_object, 'Entered the save_model method of the FileOperation class')
        path = os.path.join(self.model_directory, filename)  # create seperate directory for each cluster
        if os.path.isdir(path):  # remove previously existing models for each clusters
            shutil.rmtree(self.model_directory)
            os.makedirs(path)
        else:
            os.makedirs(path)  #
        with open(path + '/' + filename + '.sav', 'wb') as f:
            pickle.dump(model, f)  # save the model to file
        self.logger_object.log(self.file_object, 'Model File ' + filename +
                               ' saved. Exited the save_model method of the Model_Finder class')
        return 'success'

    def load_model(self, filename):
        """
        Load the saved model for predictions.
        params: filename
        Open the directory where the model is present --> load the model.
        returns: None
        """
        self.logger_object.log(self.file_object, 'Entered the load_model method of the FileOperation class')
        with open(self.model_directory + filename + '/' + filename + '.sav', 'rb') as f:
            self.logger_object.log(self.file_object, 'Model File ' + filename +
                                   ' loaded. Exited the load_model method of the Model_Finder class')
            return pickle.load(f)

    def find_correct_model_file(self, cluster_number):
        """
        Find the correct model file for the corresponding cluster.
        params: cluster_number
        Iterate over all model files in the directory --> return the correct model name
        returns: None
        """
        model_name = None
        self.logger_object.log(self.file_object,
                               'Entered the find_correct_model_file method of the FileOperation class')
        folder_name = self.model_directory
        list_of_files = os.listdir(folder_name)
        for file in list_of_files:
            try:
                if file.index(str(cluster_number)) != -1:
                    model_name = file
            except ValueError:
                continue
        model_name = model_name.split('.')[0]
        self.logger_object.log(self.file_object,
                               'Exited the find_correct_model_file method of the Model_Finder class.')
        return model_name
