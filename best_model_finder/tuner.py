from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


class ModelFinder:
    """
    This class is used to find the model with the best accuracy and AUC score.
    """

    def __init__(self, file_object, logger_object):
        # self.var_smoothing = None
        # self.grid = None
        # self.param_grid = None
        self.file_object = file_object
        self.logger_object = logger_object
        self.gnb = GaussianNB()
        self.rfc = RandomForestClassifier(random_state=41)

    def get_best_params_for_naive_bayes(self, train_x, train_y):
        """
        Gets the parameters for the Naive Bayes's Algorithm which gives the best accuracy. Hyperparameter Tuning is
        used.
        params: train_x, train_y
        First we initialize self.params_grid which is a dictionary of parameters names as keys and lists of parameter
        settings to try as values --> next we create an object of the GridSearchCv class to find the best parameters -->
        next use grifit on the training dataset to get the best parameters for the gnb model --> next extract the best
        parameters into 'self.var_smoothing' --> create a new model using the gnb algorithm with the best parameters.
        returns: self.gnb --> Gaussian Naive Bayes model with the best parameters
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        # initializing with different combination of parameters
        param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}
        # Creating an object of the Grid Search class
        grid = GridSearchCV(estimator=self.gnb, param_grid=param_grid, cv=3, verbose=3)
        # finding the best parameters
        grid.fit(train_x, train_y)
        # extracting the best parameters
        var_smoothing = grid.best_params_['var_smoothing']
        # creating a new model with the best parameters
        self.gnb = GaussianNB(var_smoothing=var_smoothing)
        # training the mew model with the best parameters on the training dataset
        self.gnb.fit(train_x, train_y)
        self.logger_object.log(self.file_object, 'Naive Bayes best params: ' + str(grid.best_params_) +
                               '. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
        return self.gnb

    def get_best_params_for_rf(self, train_x, train_y):
        """
        Gets the parameters for the random forest classifier Algorithm which gives the best accuracy. Hyperparameter
        tuning is used.
        params: train_x, train_y
        First we initialize param_grid_rfc which is a dictionary of parameters names as keys and lists of parameter
        settings to try as values --> next we create an object of the GridSearchCv class to find the best parameters -->
        next use fit on the training dataset to get the best parameters for the xgb model --> next extract the best
        parameters into 'self.var_smoothing' --> create a new model using the gnb algorithm with the best parameters.
        returns: self.rfc --> rfc model with the best parameters
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_rfc method of the Model_Finder class')
        # initializing with different combination of parameters
        param_grid_rfc = {'n_estimators': [200, 400, 500], 'max_features': ['auto', 'sqrt', 'log2'],
                          'max_depth': [3, 4, 6, 7], 'criterion': ['gini', 'entropy']}
        # Creating an object of the Grid Search class
        cv_rfc = GridSearchCV(estimator=self.rfc, param_grid=param_grid_rfc, cv=5)
        cv_rfc.fit(train_x, train_y)
        # extracting the best parameters
        n_estimators = cv_rfc.best_params_['n_estimators']
        max_features = cv_rfc.best_params_['max_features']
        max_depth = cv_rfc.best_params_['max_depth']
        criterion = cv_rfc.best_params_['criterion']
        # creating a new model with the best parameters
        self.rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          max_features=max_features, random_state=42)
        # training the mew model
        self.rfc.fit(train_x, train_y)
        self.logger_object.log(self.file_object, 'rfc best params: ' + str(cv_rfc.best_params_) +
                               '. Exited the get_best_params_for_rfc method of the Model_Finder class')
        return self.rfc

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Get the best model to use for prediction.
        params: train_x, train_y, test_x, test_y
        Get the xgboost model --> predict the test data --> if there is only one label in y, then roc_auc_score returns
        an error. We will use accuracy in that case else we use roc_auc score --> repeat the same with the naive bayes
        model. Check which score is higher and return that model.
        returns: 'model_name', model object
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_model method of the Model_Finder class')
        # create best model for rfc
        rfc = self.get_best_params_for_rf(train_x, train_y)
        prediction_rfc = rfc.predict(test_x)  # Predictions using the rfc Model
        # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
        if len(test_y.unique()) == 1:
            rfc_score = accuracy_score(test_y, prediction_rfc)
            self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(rfc_score))
        else:
            rfc_score = roc_auc_score(test_y, prediction_rfc)
            self.logger_object.log(self.file_object, 'AUC for rfc:' + str(rfc_score))
        # create best model for naive bayes
        naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
        prediction_naive_bayes = naive_bayes.predict(test_x)
        if len(test_y.unique()) == 1:
            naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
            self.logger_object.log(self.file_object, 'Accuracy for NB:' + str(naive_bayes_score))
        else:
            naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)
            self.logger_object.log(self.file_object, 'AUC for NB:' + str(naive_bayes_score))
        # comparing the two models
        if naive_bayes_score < rfc_score:
            return 'RFC', rfc
        else:
            return 'NaiveBayes', naive_bayes
