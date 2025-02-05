from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load saved models and preprocessors
rf_model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
nn_model = joblib.load('models/nn_model.pkl')

# Define prediction function
def predict_with_explanation(new_composition):
    scaled_comp = scaler.transform(new_composition)
    pred_encoded = rf_model.predict(scaled_comp)
    pred_label = label_encoder.inverse_transform(pred_encoded)
    prob_scores = rf_model.predict_proba(scaled_comp)
    
    explanation = f"Predicted Crack Susceptibility: {pred_label[0]}\n"
    for i, label in enumerate(label_encoder.classes_):
        explanation += f"{label}: {prob_scores[0][i]:.2f}\n"

    return pred_label[0], explanation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        new_composition = pd.DataFrame([data])
        prediction, explanation = predict_with_explanation(new_composition)
        return jsonify({'prediction': prediction, 'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
