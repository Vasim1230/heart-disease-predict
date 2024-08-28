from flask import Flask, render_template, jsonify, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('heart.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def man():
    return render_template('front.html')

@app.route('/predict', methods=['GET', 'POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['cp']
    
    data4 = request.form['chol']
    data5 = request.form['fbs']
    data6 = request.form['restecg']
    data7 = request.form['talach']
    data8 = request.form['exang']
    data9 = request.form['oldpeak']
    data10 = request.form['slope']
    data11 = request.form['ca']
    data12 = request.form['thal']
    data13=request.form['trestbps']
    

    arr = np.array([[data1, data2, data3, data4, data5, data6,data7,data8,data9,data10,data11,data12,data13]])
    pred = model.predict(arr)

    return render_template('home.html',data=pred, data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6,data7=data7, data8=data8, data9=data9,
                           data10=data10,data11=data11, data12=data12,data13=data13)



if __name__ == "__main__":
    app.run(debug=True)


