import joblib
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify

# Configure logging for better error visibility
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- Model and Scaler Loading ---
model = None
scaler = None

def load_models():
    """Loads the pre-trained model and scaler."""
    global model, scaler
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("✅ Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error: A required file was not found. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the same directory as app.py. Details: {e}")
        model = None
        scaler = None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model or scaler. Details: {e}")
        model = None
        scaler = None

load_models()

# --- Feature Mappings ---
SEX_MAP = {'male': 1, 'female': 0}
CP_MAP = {'typical_angina': 1, 'atypical_angina': 2, 'non_anginal': 3, 'asymptomatic': 4}
FBS_MAP = {'yes': 1, 'no': 0}
RESTECG_MAP = {'normal': 0, 'st_t_wave_abnormality': 1, 'left_ventricular_hypertrophy': 2}
EXANG_MAP = {'yes': 1, 'no': 0}
SLOPE_MAP = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
THAL_MAP = {'normal': 3, 'fixed_defect': 6, 'reversible_defect': 7}

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if model is None or scaler is None:
        logging.error("Model or scaler not loaded. Returning 500.")
        return jsonify({'error': 'Server model not ready. Please check server logs.'}), 500
    
    try:
        data = request.json
        logging.info(f"Received data for prediction: {data}")
        
        # Collect and transform features from the JSON request
        features = [
            float(data.get('age')),
            SEX_MAP.get(data.get('sex')),
            CP_MAP.get(data.get('cp')),
            float(data.get('trestbps')),
            float(data.get('chol')),
            FBS_MAP.get(data.get('fbs')),
            RESTECG_MAP.get(data.get('restecg')),
            float(data.get('thalach')),
            EXANG_MAP.get(data.get('exang')),
            float(data.get('oldpeak')),
            SLOPE_MAP.get(data.get('slope')),
            float(data.get('ca')),
            THAL_MAP.get(data.get('thal'))
        ]
        
        # Check for any missing values
        if any(f is None for f in features):
            logging.error(f"Input validation failed: Missing or invalid value.")
            return jsonify({'error': 'Invalid input data. Please ensure all fields are filled.'}), 400
        
        # Convert the features to a numpy array and reshape for the model
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data)
        
        # Make a prediction
        prediction_result = model.predict(scaled_data)[0]
        
        # Determine the human-readable result
        result_label = "Heart Disease" if prediction_result == 1 else "No Heart Disease"
        logging.info(f"Prediction result: {result_label}")
        
        return jsonify({'prediction': result_label})
    
    except Exception as e:
        logging.error(f"An error occurred during prediction. Details: {e}")
        return jsonify({'error': 'Invalid input data or an internal server error occurred.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
import joblib
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify

# Configure logging for better error visibility
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- Model and Scaler Loading ---
model = None
scaler = None

def load_models():
    """Loads the pre-trained model and scaler."""
    global model, scaler
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("✅ Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error: A required file was not found. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the same directory as app.py. Details: {e}")
        model = None
        scaler = None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model or scaler. Details: {e}")
        model = None
        scaler = None

load_models()

# --- Feature Mappings ---
SEX_MAP = {'male': 1, 'female': 0}
CP_MAP = {'typical_angina': 1, 'atypical_angina': 2, 'non_anginal': 3, 'asymptomatic': 4}
FBS_MAP = {'yes': 1, 'no': 0}
RESTECG_MAP = {'normal': 0, 'st_t_wave_abnormality': 1, 'left_ventricular_hypertrophy': 2}
EXANG_MAP = {'yes': 1, 'no': 0}
SLOPE_MAP = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
THAL_MAP = {'normal': 3, 'fixed_defect': 6, 'reversible_defect': 7}

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if model is None or scaler is None:
        logging.error("Model or scaler not loaded. Returning 500.")
        return jsonify({'error': 'Server model not ready. Please check server logs.'}), 500
    
    try:
        data = request.json
        logging.info(f"Received data for prediction: {data}")
        
        # Collect and transform features from the JSON request
        features = [
            float(data.get('age')),
            SEX_MAP.get(data.get('sex')),
            CP_MAP.get(data.get('cp')),
            float(data.get('trestbps')),
            float(data.get('chol')),
            FBS_MAP.get(data.get('fbs')),
            RESTECG_MAP.get(data.get('restecg')),
            float(data.get('thalach')),
            EXANG_MAP.get(data.get('exang')),
            float(data.get('oldpeak')),
            SLOPE_MAP.get(data.get('slope')),
            float(data.get('ca')),
            THAL_MAP.get(data.get('thal'))
        ]
        
        # Check for any missing values
        if any(f is None for f in features):
            logging.error(f"Input validation failed: Missing or invalid value.")
            return jsonify({'error': 'Invalid input data. Please ensure all fields are filled.'}), 400
        
        # Convert the features to a numpy array and reshape for the model
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data)
        
        # Make a prediction
        prediction_result = model.predict(scaled_data)[0]
        
        # Determine the human-readable result
        result_label = "Heart Disease" if prediction_result == 1 else "No Heart Disease"
        logging.info(f"Prediction result: {result_label}")
        
        return jsonify({'prediction': result_label})
    
    except Exception as e:
        logging.error(f"An error occurred during prediction. Details: {e}")
        return jsonify({'error': 'Invalid input data or an internal server error occurred.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
import joblib
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify

# Configure logging for better error visibility
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- Model and Scaler Loading ---
model = None
scaler = None

def load_models():
    """Loads the pre-trained model and scaler."""
    global model, scaler
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("✅ Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error: A required file was not found. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the same directory as app.py. Details: {e}")
        model = None
        scaler = None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model or scaler. Details: {e}")
        model = None
        scaler = None

load_models()

# --- Feature Mappings ---
SEX_MAP = {'male': 1, 'female': 0}
CP_MAP = {'typical_angina': 1, 'atypical_angina': 2, 'non_anginal': 3, 'asymptomatic': 4}
FBS_MAP = {'yes': 1, 'no': 0}
RESTECG_MAP = {'normal': 0, 'st_t_wave_abnormality': 1, 'left_ventricular_hypertrophy': 2}
EXANG_MAP = {'yes': 1, 'no': 0}
SLOPE_MAP = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
THAL_MAP = {'normal': 3, 'fixed_defect': 6, 'reversible_defect': 7}

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if model is None or scaler is None:
        logging.error("Model or scaler not loaded. Returning 500.")
        return jsonify({'error': 'Server model not ready. Please check server logs.'}), 500
    
    try:
        data = request.json
        logging.info(f"Received data for prediction: {data}")
        
        # Collect and transform features from the JSON request
        features = [
            float(data.get('age')),
            SEX_MAP.get(data.get('sex')),
            CP_MAP.get(data.get('cp')),
            float(data.get('trestbps')),
            float(data.get('chol')),
            FBS_MAP.get(data.get('fbs')),
            RESTECG_MAP.get(data.get('restecg')),
            float(data.get('thalach')),
            EXANG_MAP.get(data.get('exang')),
            float(data.get('oldpeak')),
            SLOPE_MAP.get(data.get('slope')),
            float(data.get('ca')),
            THAL_MAP.get(data.get('thal'))
        ]
        
        # Check for any missing values
        if any(f is None for f in features):
            logging.error(f"Input validation failed: Missing or invalid value.")
            return jsonify({'error': 'Invalid input data. Please ensure all fields are filled.'}), 400
        
        # Convert the features to a numpy array and reshape for the model
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data)
        
        # Make a prediction
        prediction_result = model.predict(scaled_data)[0]
        
        # Determine the human-readable result
        result_label = "Heart Disease" if prediction_result == 1 else "No Heart Disease"
        logging.info(f"Prediction result: {result_label}")
        
        return jsonify({'prediction': result_label})
    
    except Exception as e:
        logging.error(f"An error occurred during prediction. Details: {e}")
        return jsonify({'error': 'Invalid input data or an internal server error occurred.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
