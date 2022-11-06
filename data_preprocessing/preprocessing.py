import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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
        self.logger_object.log(self.file_object,'Label Separation Successful. Exited the separate_label_feature '
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
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None
                     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        self.cols_with_missing_values = cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object,
                                   'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def scale_numerical_columns(self, data):
        """
                                                        Method Name: scale_numerical_columns
                                                        Description: This method scales the numerical values using the Standard scaler.
                                                        Output: A dataframe with scaled
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None
                                     """
        self.logger_object.log(self.file_object,
                               'Entered the scale_numerical_columns method of the Preprocessor class')

        self.data = data

        try:
            self.num_df = self.data.select_dtypes(include=['int64']).copy()
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()

    def encode_categorical_columns(self, data):
        """
                                                Method Name: encode_categorical_columns
                                                Description: This method encodes the categorical values to numeric values.
                                                Output: only the columns with categorical values converted to numerical values
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.logger_object.log(self.file_object,
                               'Entered the encode_categorical_columns method of the Preprocessor class')

        try:
            self.cat_df = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numericsl ones
            for col in self.cat_df.columns:
                self.cat_df = pd.get_dummies(self.cat_df, columns=[col], prefix=[col], drop_first=True)

            self.logger_object.log(self.file_object,
                                   'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.cat_df

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def handle_imbalanced_dataset(self, x, y):
        """
        Method Name: handle_imbalanced_dataset
        Description: This method handles the imbalanced dataset to make it a balanced one.
        Output: new balanced feature and target columns
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
                                     """
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            self.rdsmple = RandomOverSampler()
            self.x_sampled, self.y_sampled = self.rdsmple.fit_sample(x, y)
            self.logger_object.log(self.file_object,
                                   'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return self.x_sampled, self.y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()
