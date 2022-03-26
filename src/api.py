import pickle
import pandas as pd
import os
import sklearn
import numpy as np
from flask import Flask, request, Response
from lightgbm import LGBMClassifier
from class_.FraudDetection import FraudDetection 


model = pickle.load(open('model/lgbm.pkl', 'rb')) # loading model
app = Flask(__name__) # initialize API

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
    porta = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=porta)


"""
import json
import requests

# data to json
data = json.dumps(x_test.to_dict(orient='records'))

#url = 'http://127.0.0.1:5000/fraudDetection/predict'
url = 'https://api-fraud.herokuapp.com/fraudDetection/predict' # local host
header = {'content-type': 'application/json'} # set type as json

# request with method POST
response = requests.post(url, data=data, headers=header)
print('Status code: {}'.format(response.status_code))

# json to dataframe
d1 = pd.DataFrame(response.json(), columns=response.json()[0].keys())
d1"""