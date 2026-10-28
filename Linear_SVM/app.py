
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        variance = float(request.form["variance"])
        skewness = float(request.form["skewness"])
        curtosis = float(request.form["curtosis"])
        entropy = float(request.form["entropy"])

        features = np.array([[
            variance,
            skewness,
            curtosis,
            entropy
        ]])

        # Scale input features
        features = scaler.transform(features)

        # Predict banknote class
        result = int(model.predict(features)[0])

        if result == 0:
            prediction = "Genuine Banknote"
        else:
            prediction = "Counterfeit Banknote"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
