import os

from prediction_Validation_Insertion import PredictionValidation
from train_model import TrainModel
from training_Validation_Insertion import TrainValidation
from predict_with_models import Prediction

from flask import Flask, request, render_template
from flask import Response
from flask_cors import CORS, cross_origin

# import flask_monitoringdashboard as dashboard

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
# dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route_client():
    try:
        if request.json is not None:
            path = request.json['filepath']
            pred_val = PredictionValidation(path)
            pred_val.prediction_validation()  # calling the prediction_validation function
            pred = Prediction(path)
            path = pred.data_prediction()  # predicting for dataset present in database
            return Response("Prediction File created at %s!!!" % path)
        elif request.form is not None:
            path = request.form['filepath']
            pred_val = PredictionValidation(path)
            pred_val.prediction_validation()  # calling the prediction_validation function
            pred = Prediction(path)
            path = pred.data_prediction()  # predicting for dataset present in database
            return Response("Prediction File created at %s!!!" % path)
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def train_route_client():
    try:
        if request.json['filepath'] is not None:
            path = request.json['filepath']
            train_val_obj = TrainValidation(path)
            train_val_obj.train_validation()  # calling the training_validation function
            train_model_obj = TrainModel()
            train_model_obj.train_model_on_data()  # training the model for the files in the table
            return Response("Training successfull!!")
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


port = int(os.getenv("PORT", 5001))
if __name__ == "__main__":
    app.run(port=port, debug=True)
