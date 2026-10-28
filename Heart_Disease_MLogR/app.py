from flask import Flask, render_template, request

import pickle

import numpy as np

app = Flask(__name__)

# Load trained model

model = pickle.load(open("HeartDiseaseModel.pkl", "rb"))

@app.route("/")

def home():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():

    # Get values from HTML form

    male = float(request.form["male"])

    age = float(request.form["age"])

    education = float(request.form["education"])

    currentSmoker = float(request.form["currentSmoker"])

    cigsPerDay = float(request.form["cigsPerDay"])

    BPMeds = float(request.form["BPMeds"])

    prevalentStroke = float(request.form["prevalentStroke"])

    prevalentHyp = float(request.form["prevalentHyp"])

    diabetes = float(request.form["diabetes"])

    totChol = float(request.form["totChol"])

    sysBP = float(request.form["sysBP"])

    diaBP = float(request.form["diaBP"])

    BMI = float(request.form["BMI"])

    heartRate = float(request.form["heartRate"])

    glucose = float(request.form["glucose"])

    # Arrange inputs in EXACT same order as training

    input_data = np.array([[

        male,

        age,

        education,

        currentSmoker,

        cigsPerDay,

        BPMeds,

        prevalentStroke,

        prevalentHyp,

        diabetes,

        totChol,

        sysBP,

        diaBP,

        BMI,

        heartRate,

        glucose

    ]])

    # Prediction

    prediction = model.predict(input_data)[0]

    # Probability

    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:

        result = "Higher predicted risk of CHD"

    else:

        result = "Lower predicted risk of CHD"

    return render_template(

        "index.html",

        prediction=result,

        probability=round(probability, 2)

    )

if __name__ == "__main__":

    app.run(debug=True)