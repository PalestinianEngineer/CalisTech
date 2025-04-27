from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load models on startup
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

def extract_features_from_window(window):
    features = {}
    for col in window.columns:
        signal = window[col].values
        features[f'{col}_mean'] = np.mean(signal)
        features[f'{col}_std'] = np.std(signal)
        features[f'{col}_min'] = np.min(signal)
        features[f'{col}_max'] = np.max(signal)
        features[f'{col}_energy'] = np.sum(signal**2) / len(signal)
        features[f'{col}_zero_crossing'] = ((signal[:-1] * signal[1:]) < 0).sum()
    return features

def sliding_window_feature_extraction(df, window_size=125, step_size=62):
    feature_rows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        if len(window) == window_size:
            features = extract_features_from_window(window)
            feature_rows.append(features)
    return pd.DataFrame(feature_rows)

def predict_from_csv(csv_file_path):
    # Load CSV
    df = pd.read_csv(csv_file_path)

    # Keep only 'ax', 'ay', 'az'
    expected_columns = ['ax', 'ay', 'az']
    df = df[[col for col in df.columns if col in expected_columns]]

    # Add 'aT' column
    df['aT'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

    # Extract features
    features_df = sliding_window_feature_extraction(df)
    
    # Standardize features
    features_scaled = scaler.transform(features_df)

    # Predict
    predictions_encoded = mlp_model.predict(features_scaled)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    return pd.Series(predictions).value_counts().to_dict()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'})

    try:
        # Save uploaded file temporarily
        temp_path = 'temp.csv'
        file.save(temp_path)
        
        # Make prediction
        results = predict_from_csv(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Exercise Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        .error {
            color: red;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Exercise Prediction</h1>
        <p>Upload a CSV file containing accelerometer data (ax, ay, az columns)</p>
        
        <form id="uploadForm">
            <input type="file" id="csvFile" accept=".csv" required>
            <button type="submit">Predict</button>
        </form>

        <div id="error" class="error"></div>
        <div id="result" class="result"></div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('csvFile');
            formData.append('file', fileInput.files[0]);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const errorDiv = document.getElementById('error');
                const resultDiv = document.getElementById('result');
                
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.style.display = 'block';
                    resultDiv.style.display = 'none';
                } else {
                    errorDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = '<h3>Predictions:</h3>';
                    
                    for (const [exercise, count] of Object.entries(data.predictions)) {
                        resultDiv.innerHTML += `<p>${exercise}: ${count}</p>`;
                    }
                }
            })
            .catch(error => {
                document.getElementById('error').textContent = 'An error occurred';
                document.getElementById('error').style.display = 'block';
                document.getElementById('result').style.display = 'none';
            });
        };
    </script>
</body>
</html>
        ''')
    
    app.run(debug=True)
