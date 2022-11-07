import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods


class KMeansClustering:
    """
    This class is used to create the optimum number of clusters of the data so that the best suited model can then
    be applied on these clusters to gather insights.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self, data):
        """
        This method analyzes the data and returns an elbow plot diagram which displays the best number of clusters.
        params: data
        Loop over numbers 1 to 11 --> find the wcss --> plot the graph between wcss and number of clusters --> save
        the plot --> get the optimal number of clusters (knee)
        returns: kn.knee
        """
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')
        wcss = list()  # initializing an empty list
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
            kmeans.fit(data)  # fitting the data to the KMeans Algorithm
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)  # creating the graph between WCSS and the number of clusters
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        # plt.show()
        plt.savefig('preprocessing_data/K-Means_Elbow.PNG')  # saving the elbow plot locally
        # finding the value of the optimum cluster programmatically
        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        self.logger_object.log(self.file_object, 'The optimum number of clusters is: ' + str(kn.knee) +
                               ' . Exited the elbow_plot method of the KMeansClustering class')
        return kn.knee

    def create_clusters(self, data, number_of_clusters):
        """
        Create a new dataframe with the cluster information of the data.
        params: data, number_of_clusters
        create an object of k means ++ clustering algorithm --> divide the data into clusters --> save the kmeans model
        --> add a cluster column in the dataframe with its cluster information.
        returns: data
        """
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(data)  # divide data into clusters
        file_op = file_methods.File_Operation(self.file_object, self.logger_object)
        file_op.save_model(kmeans, 'KMeans')  # saving the KMeans model to directory
        # passing 'Model' as the functions need three parameters
        data['Cluster'] = y_kmeans  # create a new column in dataset for storing the cluster information
        return data
