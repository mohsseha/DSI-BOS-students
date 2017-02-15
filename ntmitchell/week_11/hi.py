#!/usr/bin/python

# Import flask and create a flask object
import flask
app = flask.Flask(__name__)

#------------ MODEL ------------

# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import data
df = pd.read_csv("./titanic.csv")

# Define predictor variables
include = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Survived"]

# Clean data
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == "male" else 1)
df = df[include].dropna()

# Construct the model
X = df[["Pclass", "Sex", "Age", "Fare", "SibSp"]]
y = df["Survived"]

# Create the predictor
PREDICTOR = RandomForestClassifier(n_estimators = 100).fit(X, y)


#------------ ROUTES ------------

@app.route('/')
def hello():
    return ("""
    <body>
    <h1>Welcome!</h1>
    <h2> Hello World! </h2>
    </body>
    """)

@app.route('/greet/<name>')
def greet(name):
    return("Hello, {}".format(name))

@app.route('/predict', methods = ["GET"])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']

    item = [pclass, sex, age, fare, sibsp]
    score = PREDICTOR.predict_proba(item)
    results = {'survival rate': score[0, 1], 'fatality rate': score[0,0]}
    return flask.jsonify(results)

@app.route('/page')
def page():
    # Open the page.html file in folder
    with open ("page.html", "r") as viz_file:
        return viz_file.read()

@app.route('/result', methods = ["POST", "GET"])
def result():
    if flask.request.method == 'POST':
        inputs = flask.request.form

        pclass = inputs['pclass'][0]
        sex = inputs['sex'][0]
        age = inputs['age'][0]
        fare = inputs['fare'][0]
        sibsp = inputs['sibsp'][0]

        item = [pclass, sex, age, fare, sibsp]
        score = PREDICTOR.predict_proba(item)
        results = {'survival rate': score[0, 1], 'fatality rate': score[0,0]}
        return flask.jsonify(results)

#------------ FUNCTION ------------

if __name__ == '__main__':

    HOST = '127.0.0.1'
    PORT = 4000
    app.run(HOST, PORT)

    # access this as http://localhost:4000/predict?pclass=1&sex=0&age=18&fare=500&sibsp=1
