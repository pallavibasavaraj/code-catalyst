from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import difflib 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle


app = Flask(__name__)


users = {'kartik': 'password123',
         'pallavi': 'password123'}

model1=pickle.load(open('model.pkl','rb'))
avg=0

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def login():
    print("entered function")
    if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            if username in users and users[username] in password:
                return render_template('file.html')
            else:
                print("error happend")
                return render_template('login.html', message='Invalid username or password')
    

    



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
            int_features=[int(x) for x in request.form.values()]
            final=[np.array(int_features)]
            predicted_values = model1.predict(final)
            print(predicted_values)
            predicted_values = model1.predict(final)
            average_prediction = np.mean(predicted_values)
            avg= average_prediction
            print("Average Predicted Value:", average_prediction)
            return render_template('insight.html',predicted_values=predicted_values,average_prediction=average_prediction)
    return render_template('insight.html')

@app.route("/rec")
def rec():
    dif=avg-140
    print(dif)
    if (dif > 0):
        return "reduce consumption by"
    if (dif <0):
        return "no recommandition"
    if (dif==0):
        return "no recmmandition"

if __name__ == "__main__":
    app.run(debug=True)