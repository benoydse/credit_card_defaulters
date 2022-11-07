import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler


class Preprocessor:
    """
    This class shall  be used to clean and transform the data before training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_unwanted_spaces(self, data):
        """
        This method removes the unwanted spaces from a pandas dataframe by stripping the spaces.
        params: data
        Get the data --> apply a lambda function to strip the data if it is of datatype object --> else pass.
        returns: df_without_spaces
        """
        self.logger_object.log(self.file_object, 'Entered the remove_unwanted_spaces method of the Preprocessor class')
        df_without_spaces = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.logger_object.log(self.file_object, 'Unwanted spaces removal Successful.Exited the '
                                                 'remove_unwanted_spaces method of the Preprocessor class')
        return df_without_spaces

    def remove_columns(self, data, columns):
        """
        Description: This method removes the columns passed from a pandas dataframe.
        params: data, columns
        Get the data --> drop the necessary columns.
        Output: useful_data
        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        useful_data = data.drop(labels=columns, axis=1)  # drop the labels specified in the columns
        self.logger_object.log(self.file_object,
                               'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
        return useful_data

    def separate_label_feature(self, data, label_column_name):
        """
        This method separates the features columns from the label column.
        params: data, label_column_name
        Get the input dataframe --> drop the target column --> feature columns --> y is the target column.
        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        # drop the columns specified and separate the feature columns
        x = data.drop(labels=label_column_name, axis=1)
        y = data[label_column_name]
        self.logger_object.log(self.file_object, 'Label Separation Successful. Exited the separate_label_feature '
                                                 'method of the Preprocessor class')
        return x, y

    def is_null_present(self, data):
        """
        This method checks whether there are null values present in the pandas Dataframe or not.
        params: data

        returns: True if null values are present in the DataFrame, False if they are not present and
                returns the list of columns for which null values are present.
        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        null_present = False
        cols_with_missing_values = list()
        cols = data.columns
        null_counts = data.isna().sum()  # check for the count of null values per column
        for i in range(len(null_counts)):
            if null_counts[i] > 0:
                null_present = True
                cols_with_missing_values.append(cols[i])
        if null_present:  # write the logs to see which columns have null values
            dataframe_with_null = pd.DataFrame()
            dataframe_with_null['columns'] = data.columns
            dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
            dataframe_with_null.to_csv(
                'preprocessing_data/null_values.csv')  # storing the null column information to file
        self.logger_object.log(self.file_object, 'Finding missing values is a success.Data written to the null values '
                                                 'file. Exited the is_null_present method of the Preprocessor class')
        return null_present, cols_with_missing_values

    def impute_missing_values(self, data, cols_with_missing_values):
        """
        This method replaces all the missing values in the Dataframe using KNN imputer.
        Iterate over columns with missing values --> impute the missing values with KNN.
        returns: data
        """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        imputer = KNNImputer(n_neighbors=2)
        for col in cols_with_missing_values:
            data[col] = imputer.fit_transform(data[col])
        self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the '
                                                 'impute_missing_values method of the Preprocessor class')
        return data

    def scale_numerical_columns(self, data):
        """
        This method scales all the numerical values using the Standard scaler to scale all the numbers in the
        same range.
        params: data
        create a df of the columns with datatype as int64 --> transform the data using standard scaler --> create a
        dataframe of the scaled columns.
        returns: scaled_df
        """
        scaler = StandardScaler()
        self.logger_object.log(self.file_object, 'Entered the scale_numerical_columns method of the '
                                                 'Preprocessor class')
        num_df = data.select_dtypes(include=['int64']).copy()
        scaled_data = scaler.fit_transform(num_df)
        scaled_df = pd.DataFrame(data=scaled_data, columns=num_df.columns)
        self.logger_object.log(self.file_object, 'scaling for numerical values successful. Exited the '
                                                 'scale_numerical_columns method of the Preprocessor class')
        return scaled_df

    def encode_categorical_columns(self, data):
        """
        This method encodes the categorical values to numeric values using an encoder.
        params: data
        Get the columns with categorical datatype --> iterate over columns --> get the encoding.
        returns: cat_df
        """
        self.logger_object.log(self.file_object,
                               'Entered the encode_categorical_columns method of the Preprocessor class')
        cat_df = data.select_dtypes(include=['object']).copy()
        # Using the dummy encoding to encode the categorical columns to numerical ones
        for col in cat_df.columns:
            cat_df = pd.get_dummies(cat_df, columns=[col], prefix=[col], drop_first=True)
        self.logger_object.log(self.file_object, 'encoding for categorical values successful. Exited the '
                                                 'encode_categorical_columns method of the Preprocessor class')
        return cat_df

    def handle_imbalanced_dataset(self, x, y):
        """
        This method handles the imbalanced dataset to make it a balanced one.
        params: x, y
        Sample the x and y data.
        returns: x_sampled, y_sampled
        """
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')
        random_sampler = RandomOverSampler()
        x_sampled, y_sampled = random_sampler.fit_sample(x, y)
        self.logger_object.log(self.file_object, 'dataset balancing successful. Exited the '
                                                 'handle_imbalanced_dataset method of the Preprocessor class')
        return x_sampled, y_sampled
