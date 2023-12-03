import pandas as pd
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import shapiro# Create a histogram
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import shapiro# Create a histogram
import pickle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

#### Defining Flask App
app = Flask(__name__)

#### A function which trains the model on all the faces available in faces folder
def load_model():
    with open(r"./static/huber_regressor_model_rs_999.pkl", "rb") as input_file:
        e = pickle.load(input_file)
    return e

model = load_model()

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    return render_template('home.html')


#### This function will run when we click on Take Attendance Button

@app.route('/predict', methods=['POST'])
def predict():
        # Get user input from the form
        school = float(request.form['school'])
        sex = float(request.form['sex'])
        age = float(request.form['age'])
        address = float(request.form['address'])
        famsize = float(request.form['famsize'])
        pstatus = float(request.form['pstatus'])
        Medu = float(request.form['Medu'])
        Fedu = float(request.form['Fedu'])
        Mjob = float(request.form['Mjob'])
        Fjob = float(request.form['Fjob'])
        reason = float(request.form['reason'])
        guardian = float(request.form['guardian'])
        traveltime = float(request.form['traveltime'])
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])
        schoolsup = float(request.form['schoolsup'])
        famsup = float(request.form['famsup'])
        paid = float(request.form['paid'])
        activities = float(request.form['activities'])
        nursery = float(request.form['nursery'])
        higher = float(request.form['higher'])
        internet = float(request.form['internet'])
        romantic = float(request.form['romantic'])
        famrel = float(request.form['famrel'])
        freetime = float(request.form['freetime'])
        goout = float(request.form['goout'])
        Dalc = float(request.form['Dalc'])
        Walc = float(request.form['Walc'])
        health = float(request.form['health'])
        absences = float(request.form['absences'])
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])

        # Call your prediction function or model here
        prediction = model.predict([[school,sex,age,address,famsize,pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2]])

        # Pass the prediction to the result template
        return render_template('home.html', 
        prediction=prediction
        )


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
