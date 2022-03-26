import pickle
import numpy as np
import pandas as pd
import json
from lightgbm import LGBMClassifier
import sklearn
import inflection


class FraudDetection:
    def __init__(self):
        
        # load transformation
        self.mms_step = pickle.load(open('preparation/mms_step.pkl', 'rb'))  
        self.rs_amount = pickle.load(open('preparation/rs_amount.pkl', 'rb'))
        self.rs_oldbalance_org = pickle.load(open('preparation/rs_oldbalance_org.pkl', 'rb'))
        self.rs_newbalance_orig = pickle.load(open('preparation/rs_newbalance_orig.pkl', 'rb'))
        self.rs_oldbalance_dest = pickle.load(open('preparation/rs_oldbalance_dest.pkl', 'rb'))
        self.rs_newbalance_dest = pickle.load(open('preparation/rs_newbalance_dest.pkl', 'rb'))
        
    def cleaning(self, df):
        
        snakecase = lambda x: inflection.underscore(x) # snakecase
        df.columns = list(map(snakecase, list(df.columns)))  # parse columns to snakecase
        
        return df
        
    def feature_engineering(self, df):
        
        # const to fauture: purchasing_power_org
        AVERAGE_SALARY = 3400
        MEDIAN_SALARY = df['oldbalance_org'].median()
        
        
        # greater_50 - describe if the transaction is greater than 50% of the balance (yes or no)
        df['greater_50%'] = df[['amount', 'oldbalance_org']].apply(
            lambda x: 'yes' if x['amount'] / 2 > x['oldbalance_org'] else 'no', axis=1
        )
        
        # purchasing_power_org - describe the purchasing power of the origin transaction
        df['purchasing_power_org'] = df['oldbalance_org'].apply(
            lambda x: 'low' if x <= AVERAGE_SALARY else 'average' 
                            if x > AVERAGE_SALARY and x <= MEDIAN_SALARY else 'high'
        )
        
        # type_transaction - describe the type of transaction C = Customer and M = Merchants
        df['type_transaction'] = df[['name_orig', 'name_dest']].apply(
            lambda x: 'C to C' if x['name_orig'][0] == 'C' and x['name_dest'][0] == 'C' else 'M to M' 
                               if x['name_orig'][0] == 'M' and x['name_dest'][0] == 'M' else 'C to M' 
                               if x['name_orig'][0] == 'C' and x['name_dest'][0] == 'M' else 'M to C' 
                               if x['name_orig'][0] == 'M' and x['name_dest'][0] == 'C' else None, axis=1
        )
        
        # type_amount - Describe is amount is greater than 80K (greater or less)
        df['type_amount'] = df['amount'].apply(lambda x: 'greater' if x > 80000 else 'less')
        
        return df
        
        
    def preparation(self, df):
            
        # apply label encoding
        df['purchasing_power_org'] = df['purchasing_power_org'].map({'high': 3, 'average': 2, 'low': 1}) # purchasing_power_org
        df['greater_50%'] = df['greater_50%'].map({'no': 0, 'yes': 1})       # greater_50
        df['type_amount'] = df['type_amount'].map({'less': 0, 'greater': 1}) # type_amount
        
        # apply one hot encoding
        df['type_PAYMENT']  = df['type'].apply(lambda x: 1 if x == 'PAYMENT' else 0)
        df['type_TRANSFER'] = df['type'].apply(lambda x: 1 if x == 'TRANSFER' else 0)
        df['type_CASH_OUT'] = df['type'].apply(lambda x: 1 if x == 'CASH_OUT' else 0)
        df['type_DEBIT']    = df['type'].apply(lambda x: 1 if x == 'DEBIT' else 0)
        df['type_CASH_IN']  = df['type'].apply(lambda x: 1 if x == 'CASH_IN' else 0)
        #df = pd.get_dummies(df, prefix='type', columns=['type']) # type
        #df = pd.get_dummies(df, prefix='type_transaction', columns=['type_transaction']) # type_transaction
        
        # apply min max scaler
        df['step'] = self.mms_step.transform(df[['step']].values) #  step
        
        # apply robust scaler
        df['amount'] = self.rs_amount.transform(df[['amount']].values) # amount
        df['oldbalance_org'] = self.rs_oldbalance_org.transform(df[['oldbalance_org']].values)    # oldbalance_org
        df['newbalance_orig'] = self.rs_newbalance_orig.transform(df[['newbalance_orig']].values) # newbalance_orig
        df['oldbalance_dest'] = self.rs_oldbalance_dest.transform(df[['oldbalance_dest']].values) # oldbalance_dest
        df['newbalance_dest'] = self.rs_newbalance_dest.transform(df[['newbalance_dest']].values) # newbalance_dest
    
        return df

    def feature_selection(self, df):

        # columns selected to model predict
        cols_selected = ['oldbalance_org', 'newbalance_orig', 'greater_50%', 'type_TRANSFER',
                         'type_amount', 'type_CASH_OUT', 'type_CASH_IN', 'purchasing_power_org',
                         'is_flagged_fraud']

        # filtrer columns in dataframe
        df = df[cols_selected]

        return df
    
    def get_prediction(self, model, original_data, test_data):

        # model prediction
        pred = model.predict(test_data)

        # Join prediction into original data
        original_data['fraud'] = list(pred)

        # original data with prediction to json
        return original_data.to_json(orient='records', date_format='iso')