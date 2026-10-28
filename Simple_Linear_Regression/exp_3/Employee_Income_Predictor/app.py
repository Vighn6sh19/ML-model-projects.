from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Multiple Linear Regression model
model = pickle.load(open("EmployeeIncomeModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    experience = float(request.form['experience'])

    prediction = model.predict(
        np.array([[age, experience]])
    )

    return render_template(
        "index.html",
        prediction_text=f"Predicted Income: ₹{prediction[0]:,.2f}"
    )


if __name__ == '__main__':
    app.run(debug=True)