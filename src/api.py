import pickle
import pandas as pd
import os
import sklearn
import numpy as np
from flask import Flask, request, Response
from lightgbm import LGBMClassifier
from class_.FraudDetection import FraudDetection


# loading model
model = pickle.load(open('C:/Users/Jhonatans/projects/ML/Classification/Financial-Fraud-Detection/src/model/lgbm.pkl', 'rb'))

# initialize API
app = Flask(__name__)

@app.route('/fraudDetection/predict', methods=['POST'])
def fraudDetection_predict():
    
    test_json = request.get_json()
    if test_json: # there is data

        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # instantiate class
        detection = FraudDetection()
        
        # data cleaning
        df1 = detection.cleaning(df=test_raw)
        print('cleaning OK')
        
        # feature engineering
        df2 = detection.feature_engineering(df=df1)
        print('feature engineering OK')
        
        # data preparation
        df3 = detection.preparation(df=df2)
        print('data preparation OK')

        # feature selection
        df4 = detection.feature_selection(df=df3)
        print('feature selection OK')
        
        # prediction
        df_response = detection.get_prediction(
            model=model, original_data=df1, test_data=df4
        )
        print('prediction OK')

        return df_response
        
    else:
        return Response('{}', status=200, minetype='application/json')
    
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=True)