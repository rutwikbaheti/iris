from crypt import methods
import pandas as pd
import numpy as pd
import joblib
import traceback
from flask import Flask, request, jsonify

app=Flask(__name__)

lr=joblib.load("model.pkl")

@app.route("/",methods=["GET"])
def info():
	return "iris flower classification"
	
@app.route("/predict", methods=["GET"])
def predict():
	json_ = request.json
	prediction=list(lr.predict(json_))
	return jsonify({"Prediction ": str(prediction)})	

if __name__ == "__main__":
	app.run(debug= True)