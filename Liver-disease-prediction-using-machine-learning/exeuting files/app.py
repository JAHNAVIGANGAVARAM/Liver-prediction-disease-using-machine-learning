from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model_path = 'SVC_liver_analysis.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        gender = request.form.get('gender')
        age = float(request.form.get('age'))
        total_bilirubin = float(request.form.get('total_bilirubin'))
        direct_bilirubin = float(request.form.get('direct_bilirubin'))
        alkaline_phosphotase = float(request.form.get('alkaline_phosphotase'))
        alamine_aminotransferase = float(request.form.get('alamine_aminotransferase'))
        aspartate_aminotransferase = float(request.form.get('aspartate_aminotransferase'))
        total_proteins = float(request.form.get('total_proteins'))
        albumin = float(request.form.get('albumin'))
        albumin_and_globulin_ratio = float(request.form.get('albumin_and_globulin_ratio'))

        # Encode gender
        gender_encoded = 1 if gender == 'Male' else 0

        # Create feature array for prediction
        features = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                              alamine_aminotransferase, aspartate_aminotransferase, total_proteins, 
                              albumin, albumin_and_globulin_ratio]])

        # Predict using the loaded model
        prediction = model.predict(features)[0]
        prediction_text = "Liver disease detected" if prediction == 1 else "No liver disease detected"

        return jsonify({'prediction_result': prediction_text})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'prediction_result': 'An error occurred during prediction.'})

if __name__ == '__main__':
    app.run(debug=True)
