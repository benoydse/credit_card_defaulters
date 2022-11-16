from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
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
        self.xgb = XGBClassifier(objective='binary:logistic', n_jobs=-1)

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

    def get_best_params_for_xgboost(self, train_x, train_y):
        """
        Gets the parameters for the xgboost Algorithm which gives the best accuracy. Hyperparameter Tuning is used.
        params: train_x, train_y
        First we initialize param_grid_xgboost which is a dictionary of parameters names as keys and lists of parameter
        settings to try as values --> next we create an object of the GridSearchCv class to find the best parameters -->
        next use fit on the training dataset to get the best parameters for the xgb model --> next extract the best
        parameters into 'self.var_smoothing' --> create a new model using the gnb algorithm with the best parameters.
        returns: self.xgb --> xgboost model with the best parameters
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        # initializing with different combination of parameters
        param_grid_xgboost = {"n_estimators": [50, 100, 130], "max_depth": range(3, 11, 1),
                              "random_state": [0, 50, 100]}
        # Creating an object of the Grid Search class
        grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3, cv=2, n_jobs=-1)
        # finding the best parameters
        grid.fit(train_x, train_y)
        # extracting the best parameters
        random_state = grid.best_params_['random_state']
        max_depth = grid.best_params_['max_depth']
        n_estimators = grid.best_params_['n_estimators']
        # creating a new model with the best parameters
        self.xgb = XGBClassifier(random_state=random_state, max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1)
        # training the mew model
        self.xgb.fit(train_x, train_y)
        self.logger_object.log(self.file_object, 'XGBoost best params: ' + str(grid.best_params_) +
                               '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
        return self.xgb

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
        # create best model for XGBoost
        xgboost = self.get_best_params_for_xgboost(train_x, train_y)
        prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model
        # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
        if len(test_y.unique()) == 1:
            xgboost_score = accuracy_score(test_y, prediction_xgboost)
            self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(xgboost_score))
        else:
            xgboost_score = roc_auc_score(test_y, prediction_xgboost)
            self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(xgboost_score))
            # create best model for naive bayes
        naive_bayes = self.get_best_params_for_naive_bayes(train_x, train_y)
        prediction_naive_bayes = naive_bayes.predict(test_x)
        if len(test_y.unique()) == 1:
            naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
            self.logger_object.log(self.file_object, 'Accuracy for NB:' + str(naive_bayes_score))
        else:
            naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)
            self.logger_object.log(self.file_object, 'AUC for RF:' + str(naive_bayes_score))
        # comparing the two models
        if naive_bayes_score < xgboost_score:
            return 'XGBoost', xgboost
        else:
            return 'NaiveBayes', naive_bayes
