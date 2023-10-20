from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('pkl files/optimized_rf.pkl')
scaler = joblib.load('pkl files/scaler.pkl')
gender_encoder = joblib.load('pkl files/gender_encoder.pkl')
location_encoder = joblib.load('pkl files/location_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle gender
    gender_input = request.form.get('gender', 'DefaultGenderValue')
    if gender_input in gender_encoder.classes_:
        gender = gender_encoder.transform([gender_input])[0]
    else:
        gender = -1  

    # Handle location
    location_input = request.form.get('location', 'DefaultLocationValue')
    if location_input in location_encoder.classes_:
        location = location_encoder.transform([location_input])[0]
    else:
        location = -1 

    # Extract data from form
    age = float(request.form['age'])
    subscription_length = float(request.form['subscription_length'])
    monthly_bill = float(request.form['monthly_bill'])
    total_usage = float(request.form['total_usage'])

    # Create a numpy array based on the extracted data
    data = np.array([[age, gender, location, subscription_length, monthly_bill, total_usage]])

    # Preprocess the data
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)

    # Convert prediction to string
    if prediction[0] == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
