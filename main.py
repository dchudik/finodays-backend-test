from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from pickle import load
from catboost import CatBoostRegressor
import numpy as np

model_cb = CatBoostRegressor().load_model('Cars_model')

with open('labels_enc.pkl', 'rb') as f:
    labels_encoding = load(f)
cat_columnts = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'drive', 'type',
                'paint_color', 'state']

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
def calculate():
    year = int(request.args.get('year')) - 1900

    manufacturer = labels_encoding['manufacturer'].transform([request.args.get('manufacturer')])
    model = labels_encoding['model'].transform([request.args.get('model')])
    condition = labels_encoding['condition'].transform([request.args.get('condition')])
    cylinders = labels_encoding['cylinders'].transform([request.args.get('cylinders')])
    fuel = labels_encoding['fuel'].transform([request.args.get('fuel')])
    odometer = int(request.args.get('odometer'))
    transmission = labels_encoding['transmission'].transform([request.args.get('transmission')])
    drive = labels_encoding['drive'].transform([request.args.get('drive')])
    car_type = labels_encoding['type'].transform([request.args.get('type')])
    paint_color = labels_encoding['paint_color'].transform([request.args.get('paint_color')])
    state = labels_encoding['state'].transform([request.args.get('state')])


    row = np.array([year, manufacturer, model, condition, cylinders, fuel, odometer, transmission, drive,
                    car_type, paint_color, state])

    price = model_cb.predict(row)

    response = jsonify(price=str(int(price)))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)