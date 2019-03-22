import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


rfc = pickle.load(open('iris_rfc.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Api - Predict Iris Dataset"

@app.route('/api', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)
    predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
    predict_request = np.array(predict_request)
    predict_request = predict_request.reshape(1, -1)
    predictions = rfc.predict(predict_request)

    return jsonify(result=predictions.tolist())

if __name__ == '__main__':
    app.run(port=8080, debug=True)
