from flask import Flask, render_template, request, redirect, url_for, flash
import requests
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

NODE_BACKEND_URL = 'http://localhost:5000/api/farmers/'


def load_model(model_path):
    """Load the trained model from a file."""
    return joblib.load(model_path)

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

def predict_soil_type(model, data):
    """Predict the soil type based on the input data."""
    return model.predict(data)

def predict_yield(model_path, input_values):
    """Predict crop yield based on the input values."""
    model = load_model(model_path)
    input_df = pd.DataFrame([input_values])
    predicted_yield = model.predict(input_df)
    return predicted_yield[0]

def predict_price(model_path, input_values):
    """Predict crop price based on the input values."""
    model = load_model(model_path)
    input_df = pd.DataFrame([input_values])
    predicted_price = model.predict(input_df)
    return predicted_price[0]

def predict_crop_type(model, data):
    """Predict the crop type based on the input data."""
    return model.predict(data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/soil_analysis', methods=['GET', 'POST'])
def soil_analysis():
    soil_model_path = 'models/soil_classifier_model.pkl'
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Load the fitted scaler
    scaler = joblib.load('models/scaler_soil_analysis.pkl')

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
        
        # Prepare data and predict soil type
        prepared_data = prepare_data(input_data, feature_names, scaler)
        model = load_model(soil_model_path)
        predicted_soil_type = predict_soil_type(model, prepared_data)
        return render_template('soil_analysis.html', predicted_soil_type=predicted_soil_type)
    
    return render_template('soil_analysis.html', predicted_soil_type=None)

@app.route('/yield_production', methods=['GET', 'POST'])
def yield_production():
    yield_model_path = 'models/crop_yield_model.pkl'
    
    if request.method == 'POST':
        input_values = {
            'Rain Fall (mm)': float(request.form['rain_fall']),
            'Fertilizer': float(request.form['fertilizer']),
            'Temperature': float(request.form['temperature']),
            'Nitrogen (N)': float(request.form['nitrogen']),
            'Phosphorus (P)': float(request.form['phosphorus']),
            'Potassium (K)': float(request.form['potassium'])
        }
        
        # Predict crop yield
        predicted_yield = predict_yield(yield_model_path, input_values)
        return render_template('yield_production.html', predicted_yield=predicted_yield)
    
    return render_template('yield_production.html', predicted_yield=None)

@app.route('/cropPrediction', methods=['GET', 'POST'])
def crop_prediction():
    model_path = 'models/crop_prediction_model.pkl'
    feature_names = ['Nitrogen (N)', 'Potassium (K)', 'Phosphate (P)', 'Temperature', 'Humidity', 'Rainfall']
    
    # Load the fitted scaler
    scaler = joblib.load('models/scaler_crop_prediction.pkl')

    if request.method == 'POST':
        input_data = {
            'Nitrogen (N)': float(request.form['nitrogen']),
            'Potassium (K)': float(request.form['potassium']),
            'Phosphate (P)': float(request.form['phosphate']),
            'Temperature': float(request.form['temperature']),
            'Humidity': float(request.form['humidity']),
            'Rainfall': float(request.form['rainfall'])
        }
        
        # Prepare data for prediction
        prepared_data = prepare_data(input_data, feature_names, scaler)
        model = load_model(model_path)
        predicted_crop = predict_crop_type(model, prepared_data)
        return render_template('cropPrediction.html', predicted_crop=predicted_crop[0])
    
    return render_template('cropPrediction.html', predicted_crop=None)

@app.route('/price_prediction', methods=['GET', 'POST'])
def price_prediction():
    # Correct paths to the model and scaler
    price_model_path = 'models/crop_price_model.pkl'
    scaler_path = 'models/standard_scaler.pkl'
    
    # Load the scaler once (could be done in a separate function or as a global variable)
    scaler = joblib.load(scaler_path)

    if request.method == 'POST':
        # Extract input values from the form
        input_values = {
            'year': float(request.form['year']),
            'month': float(request.form['month']),
            'min_price': float(request.form['min_price']),
            'max_price': float(request.form['max_price'])
        }
        
        # Prepare data for prediction
        feature_names = ['year', 'month', 'min_price', 'max_price']
        prepared_data = prepare_data(input_values, feature_names, scaler)
        
        # Load the price prediction model and make a prediction
        model = load_model(price_model_path)
        predicted_price = predict_price(price_model_path, input_values)
        
        return render_template('price_prediction.html', predicted_price=predicted_price)
    
    return render_template('price_prediction.html', predicted_price=None)


@app.route('/translate', methods=['GET'])
def translate():
    return render_template('translate.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        response = requests.post(f'{NODE_BACKEND_URL}/signup', json={'username': username, 'password': password})
        
        if response.status_code == 201:
            flash('Signup successful. Please log in.')
            return redirect(url_for('login'))
        else:
            flash('Signup failed. Please try again.')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        response = requests.post(f'{NODE_BACKEND_URL}/login', json={'username': username, 'password': password})
        
        if response.status_code == 200:
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your credentials.')
            return redirect(url_for('login'))
    
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True, port=5003)
