from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging.logger import AppLogger


class TrainModel:
    """
    This is the Entry point for training the Machine Learning Model.
    """

    def __init__(self):
        self.log_writer = AppLogger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def train_model_on_data(self):
        """
        Perform all the steps required for training the model on the data.
        params: None
        Get the data from the source --> separate features and label columns --> check for null values --> if null
        is present --> impute the missing values --> create optimal number of clusters from the data --> add a new
        column to the dataset which contains the cluster number --> iterate over clusters --> get all the data
        corresponding to the current cluster --> split the dataset into train and test --> scale the numerical
        columns --> find the best model for that cluster data --> store the model with cluster number.
        returns: None
        """
        self.log_writer.log(self.file_object, 'Start of Training')
        # First get the data from the source
        data_getter = data_loader.DataGetter(self.file_object, self.log_writer)
        data = data_getter.get_data()
        # Now perform the data preprocessing steps
        preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
        # data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation
        # create separate features and labels
        x, y = preprocessor.separate_label_feature(data, label_column_name='default payment next month')
        # check if missing values are present in the dataset
        is_null_present, cols_with_missing_values = preprocessor.is_null_present(x)
        # if missing values are there, replace them appropriately.
        if is_null_present:
            x = preprocessor.impute_missing_values(x, cols_with_missing_values)  # impute the missing values
        # now apply the clustering algorithm on the data
        kmeans = clustering.KMeansClustering(self.file_object, self.log_writer)  # initialize kmeans model object
        number_of_clusters = kmeans.elbow_plot(x)  # using the elbow plot to find the number of optimum clusters
        # Divide the data into clusters
        x = kmeans.create_clusters(x, number_of_clusters)
        # create a new column in the dataset consisting of the corresponding cluster number.
        x['Labels'] = y
        # getting the unique clusters from our dataset
        list_of_clusters = x['Cluster'].unique()
        # iterate over all the clusters and looking for the best ML algorithm to fit for each individual cluster
        for i in list_of_clusters:
            cluster_data = x[x['Cluster'] == i]  # filter the data for each cluster
            # Prepare the feature and Label columns
            cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
            cluster_label = cluster_data['Labels']
            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                random_state=355)
            train_x = preprocessor.scale_numerical_columns(x_train)
            test_x = preprocessor.scale_numerical_columns(x_test)
            model_finder = tuner.ModelFinder(self.file_object, self.log_writer)  # object initialization
            # getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(train_x, y_train, test_x, y_test)
            # saving the best model to the directory.
            file_op = file_methods.FileOperation(self.file_object, self.log_writer)
            file_op.save_model(best_model, best_model_name + str(i))
        # logging the successful Training
        self.log_writer.log(self.file_object, 'Successful End of Training')
        self.file_object.close()
