from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file 'model.pkl' not found."
    )

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetes_pedigree = float(request.form["diabetes_pedigree"])
        age = float(request.form["age"])

        # Create input array
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            probability = probabilities[1] * 100

        if prediction == 1:
            result = "Diabetes Detected"
            message = "The model predicts a higher likelihood of diabetes."
        else:
            result = "No Diabetes Detected"
            message = "The model predicts a lower likelihood of diabetes."

        return render_template(
            "index.html",
            prediction=result,
            message=message,
            probability=probability
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numerical values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)