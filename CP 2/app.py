from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the XGBoost model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input values from form
            features = [
                float(request.form["ph"]),
                float(request.form["Hardness"]),
                float(request.form["Solids"]),
                float(request.form["Chloramines"]),
                float(request.form["Sulfate"]),
                float(request.form["Conductivity"]),
                float(request.form["Organic_carbon"]),
                float(request.form["Trihalomethanes"]),
                float(request.form["Turbidity"]),
            ]
            input_array = np.array([features]).reshape(1, -1)

            # Make prediction
            prediction = model.predict(input_array)[0]

            return render_template("predict.html", prediction=int(prediction))

        except Exception as e:
            return f"Error: {e}"

    return render_template("predict.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
