from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import pickle
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

# Define paths to models and scalers
MODELS_PATH = 'models/'
SOIL_MODEL_PATH = MODELS_PATH + 'soil_classifier_model.pkl'
CROP_YIELD_MODEL_PATH = MODELS_PATH + 'crop_yield_model.pkl'
CROP_PREDICTION_MODEL_PATH = MODELS_PATH + 'crop_prediction_model.pkl'
PRICE_MODEL_PATH = MODELS_PATH + 'xgb_model.pkl'

SCALER_SOIL_PATH = MODELS_PATH + 'scaler_soil_analysis.pkl'
SCALER_YIELD_PATH = MODELS_PATH + 'scaler.pkl'
SCALER_PREDICTION_PATH = MODELS_PATH + 'standard_scaler.pkl'
SCALER_PRICE_PATH = MODELS_PATH + 'scaler_prediction_price.pkl'


def load_model(model_path):
    """Load the trained model from a file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def load_scaler(scaler_path):
    """Load the scaler from a file."""
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler


def prepare_data(input_data, feature_names, scaler=None):
    """Prepare input data for prediction."""
    df = pd.DataFrame([input_data], columns=feature_names)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)
    
    # Standardize features if scaler is provided
    if scaler:
        df = scaler.transform(df)
    
    return df


def predict(model, data):
    """Predict using the provided model and data."""
    return model.predict(data)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/soil_analysis', methods=['GET', 'POST'])
def soil_analysis():
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    scaler = load_scaler(SCALER_SOIL_PATH)
    
    if request.method == 'POST':
        input_data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        prepared_data = prepare_data(input_data, feature_names, scaler)
        model = load_model(SOIL_MODEL_PATH)
        predicted_soil_type = predict(model, prepared_data)
        return render_template('soil_analysis.html', predicted_soil_type=predicted_soil_type[0])
    
    return render_template('soil_analysis.html', predicted_soil_type=None)


@app.route('/yield_production', methods=['GET', 'POST'])
def yield_production():
    feature_names = ['Rain Fall (mm)', 'Fertilizer', 'Temperature', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    scaler = load_scaler(SCALER_YIELD_PATH)
    
    if request.method == 'POST':
        input_values = {
            'Rain Fall (mm)': float(request.form['rain_fall']),
            'Fertilizer': float(request.form['fertilizer']),
            'Temperature': float(request.form['temperature']),
            'Nitrogen (N)': float(request.form['nitrogen']),
            'Phosphorus (P)': float(request.form['phosphorus']),
            'Potassium (K)': float(request.form['potassium'])
        }

        prepared_data = prepare_data(input_values, feature_names, scaler)
        model = load_model(CROP_YIELD_MODEL_PATH)
        predicted_yield = predict(model, prepared_data)
        
        return render_template('yield_production.html', predicted_yield=predicted_yield[0])
    
    return render_template('yield_production.html', predicted_yield=None)


@app.route('/cropPrediction', methods=['GET', 'POST'])
def crop_prediction():
    feature_names = ['Nitrogen (N)', 'Potassium (K)', 'Phosphate (P)', 'Temperature', 'Humidity', 'Rainfall']
    scaler = load_scaler(SCALER_PREDICTION_PATH)
    
    if request.method == 'POST':
        input_data = {
            'Nitrogen (N)': float(request.form['nitrogen']),
            'Potassium (K)': float(request.form['potassium']),
            'Phosphate (P)': float(request.form['phosphate']),
            'Temperature': float(request.form['temperature']),
            'Humidity': float(request.form['humidity']),
            'Rainfall': float(request.form['rainfall'])
        }
        
        prepared_data = prepare_data(input_data, feature_names, scaler)
        model = load_model(CROP_PREDICTION_MODEL_PATH)
        predicted_crop = predict(model, prepared_data)
        
        return render_template('cropPrediction.html', predicted_crop=predicted_crop[0])
    
    return render_template('cropPrediction.html', predicted_crop=None)


@app.route('/price_prediction', methods=['GET', 'POST'])
def price_prediction():
    scaler = load_scaler(SCALER_PRICE_PATH)
    
    if request.method == 'POST':
        input_values = {
            'year': float(request.form['year']),
            'month': float(request.form['month']),
            'min_price': float(request.form['min_price']),
            'max_price': float(request.form['max_price'])
        }
        
        feature_names = ['year', 'month', 'min_price', 'max_price']
        prepared_data = prepare_data(input_values, feature_names, scaler)
        
        model = load_model(PRICE_MODEL_PATH)
        predicted_price = predict(model, prepared_data)
        
        return render_template('price_prediction.html', predicted_price=predicted_price[0])
    
    return render_template('price_prediction.html', predicted_price=None)


@app.route('/translate', methods=['GET'])
def translate():
    return render_template('translate.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        flash('Signup successful. Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        flash('Login successful!')
        return redirect(url_for('home'))
    
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True, port=5003)
