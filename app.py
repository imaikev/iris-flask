import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request , socket

#Carga el modelo cuando inicia
rfc = pickle.load(open('iris_rfc.pkl', 'rb'))
app = Flask(__name__)

#Define las rutas/funciones de la app
@app.route('/')
def home():
    return "API - Iris Dataset"

@app.route('/api', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)
    predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
    predict_request = np.array(predict_request)
    predict_request = predict_request.reshape(1, -1)
    predictions = rfc.predict(predict_request)
          
    #return jsonify(categoria=predictions.tolist())
    return jsonify(categoria=predictions.tolist()) , 'server': socket.gethostname())

# Inicia el web service en el puerto 8080 y habilita multi hilo
if __name__ == '__main__':
    app.run (host="0.0.0.0", port= 8080,threaded=True)
