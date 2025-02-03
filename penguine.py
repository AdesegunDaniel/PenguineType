from flask import Flask, request, jsonify, render_template, session
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier




scaler_list = joblib.load('model\penguine_scalerlist.pkl')
scaler_yr = joblib.load('model\penguin_scaler_year.pkl')
model=joblib.load('model\PenguineType.model.pkl')

def preprocess(xtest, scaler_list, scaler_yr):
    for i,scaler in enumerate(scaler_list, start=1):
        xtest.iloc[:,i]=scaler.transform(xtest.iloc[:,i].values.reshape(-1,1)).reshape(-1,)
    xtest.iloc[:,6]=scaler_yr.transform(xtest.iloc[:,6].values.reshape(-1,1)).reshape(-1,)    
    return xtest

decode={0: 'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}


app = Flask(__name__)
app.secret_key =os.urandom(24)

@app.route('/') 
def pg_home():
    return render_template('pg_home.html')



@app.route('/predict', methods=['POST'])
def predict():
    island=request.form['island']
    bill_length_mm=request.form['bill_length_mm']
    bill_depth_mm=request.form['bill_depth_mm']
    flipper_length_mm=request.form['flipper_length_mm']
    body_mass_g=request.form['body_mass_g']
    sex=request.form['sex']
    year=request.form['year']

    info = {
        'island': island,
        'bill_length_mm': float(bill_length_mm),
        'bill_depth_mm': float(bill_depth_mm),
        'flipper_length_mm': float(flipper_length_mm),
        'body_mass_g': float(body_mass_g),
        'sex': sex,
        'year': int(year)
    }
    
    info=pd.DataFrame(info, index=[0])
    info['island']=info['island'].astype('category')
    info['sex']=info['sex'].astype('category')
    
    predictor= preprocess(info, scaler_list, scaler_yr)
    
    prediction=model.predict(predictor)
    prediction= decode[np.argmax(prediction)]
    return render_template('pg_result.html',
                           prediction=prediction,
                           island=island,
                           bill_length_mm=bill_length_mm,
                           bill_depth_mm=bill_depth_mm,
                           flipper_length_mm=flipper_length_mm,
                           body_mass_g=body_mass_g,
                           sex=sex,
                           year=year)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=82)

