import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler 
from src.logger import logging  

app = Flask(__name__)

@app.route('/')
def home():
    logging.info("Home page accessed.")
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info("Rendering prediction home.html form via GET request.")
        return render_template('home.html')
    else:
        logging.info("Received POST request for prediction. Extracting data from form...")
        
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        logging.info("Converting form data into Pandas DataFrame...")
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        logging.info("Initiating PredictPipeline to get prediction results...")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(features=pred_df)
        
        logging.info(f"Prediction successful! Result: {results[0]}")
        
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', debug=True)