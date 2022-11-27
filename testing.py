from prediction_Validation_Insertion import PredictionValidation
from train_model import TrainModel
from training_Validation_Insertion import TrainValidation
from predict_with_models import Prediction

# training
path = 'D:\DS_projects\credit_card_defaulters\Training_Batch_Files'
train_val_obj = TrainValidation(path)
train_val_obj.train_validation()  # calling the training_validation function
train_model_obj = TrainModel()
train_model_obj.train_model_on_data()  # training the model for the files in the table


#prediction
path = 'D:\DS_projects\credit_card_defaulters\Prediction_Batch_files'
pred_val = PredictionValidation(path)
pred_val.prediction_validation()  # calling the prediction_validation function
pred = Prediction(path)
path = pred.data_prediction()  # predicting for dataset present in database

