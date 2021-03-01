import json

import joblib
import numpy as np
from flask import Flask, request, jsonify

file_name = "resources/price-model.pkl"
model = joblib.load(file_name)
app = Flask(__name__)


@app.route("/")
def index():
    return json.dumps({"answer": "House Price Prediction"})


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)["data"]
    print("Received Data", data)
    data = np.array(data)
    print("Is model null : ", model is None)
    result = model.predict([data])[0]
    return jsonify({"answer": round(result, 4)})


if __name__ == '__main__':
    print("Flask Application Started along With Linear Regression Model")
    app.run(host="0.0.0.0", port=8080)
