from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your saved model and encoders
model = joblib.load("churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Define categorical options for dropdowns
categorical_options = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_columns=feature_columns,
        categorical_options=categorical_options,
    )

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    for col in feature_columns:
        val = request.form[col]
        if col in categorical_options:
            # Use label encoder for categorical data
            if col in label_encoders:
                val = label_encoders[col].transform([val])[0]
            else:
                # If no encoder, just use the raw value (rare)
                val = val
        else:
            # Convert numerical inputs to float
            try:
                val = float(val)
            except:
                val = 0.0
        input_data.append(val)

    prediction = model.predict([input_data])[0]
    result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Stay ✅"
    return render_template(
        "index.html",
        prediction_text=result,
        feature_columns=feature_columns,
        categorical_options=categorical_options,
    )

if __name__ == "__main__":
    app.run(debug=True)
