from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import pickle
from sklearn.impute import SimpleImputer
import os
from google.generativeai import GenerativeModel, upload_file, configure
import requests
import easyocr
import io
import re
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, render_template, request, flash
import easyocr
import io
import re
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import numpy as np

app = Flask(__name__)

# Initialize Generative Model
api_key = 'AIzaSyDptjxEz5zOW22GJikdaYzA_w_8PAIIiPI'  # Replace with your actual API key
configure(api_key=api_key)
model = GenerativeModel("gemini-1.5-flash")

# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

def format_text(text):
    """Format the text for HTML rendering."""
    import re

    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<h3>\1</h3>', text)
    formatted_text = re.sub(r'\*(.*?)\*', r'<p><i>\1</i></p>', formatted_text)

    return formatted_text

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

@app.route('/disease', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the uploaded file
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            
            # Upload the file and generate content
            myfile = upload_file(file_path)
            result = model.generate_content(
                [myfile, "\n\n", "Identify the crop and any diseases present in this image. Provide a detailed description including symptoms, diagnosis, and suggested treatments."]
            )
            # Format the result
            formatted_result = format_text(result.text)
            return render_template('disease.html', result=formatted_result)
    
    return render_template('disease.html', result=None)

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    forecast = None
    error = None
    if request.method == 'POST':
        city = request.form.get('city')
        if not city:
            error = 'Please enter a city name.'
        else:
            try:
                # OpenWeatherMap API Key and URL with placeholders for city
                api_key = '2e510a7933d5c316e5bec3f8ef4d259b'  # Replace with your actual API key
                url = f'https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric'
                
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    if 'list' in data:
                        forecast_data = data['list']
                        forecast = {
                            'city': data['city']['name'],
                            'days': []
                        }
                        
                        # Aggregate forecast by day
                        for item in forecast_data:
                            date = item['dt_txt'].split(' ')[0]  # Extract date part from 'dt_txt'
                            day_forecast = next((day for day in forecast['days'] if day['date'] == date), None)
                            
                            if not day_forecast:
                                day_forecast = {
                                    'date': date,
                                    'maxtemp_c': item['main']['temp_max'],
                                    'mintemp_c': item['main']['temp_min'],
                                    'condition': item['weather'][0]['description']
                                }
                                forecast['days'].append(day_forecast)
                            else:
                                day_forecast['maxtemp_c'] = max(day_forecast['maxtemp_c'], item['main']['temp_max'])
                                day_forecast['mintemp_c'] = min(day_forecast['mintemp_c'], item['main']['temp_min'])

                    else:
                        error = 'City not found or invalid response format.'
                else:
                    error = f'API error with status code {response.status_code}. Message: {data.get("message", "Unknown error")}'
            except Exception as e:
                error = f'An error occurred: {str(e)}'
    
    return render_template('weather.html', forecast=forecast, error=error)

reader = easyocr.Reader(['en'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    
    # Apply a filter to reduce noise
    image = image.filter(ImageFilter.SHARPEN)
    
    return image

def extract_text_from_image(image_file):
    try:
        # Read the image file as a binary stream
        image_bytes = io.BytesIO(image_file.read())
        
        # Open the image with PIL
        image = Image.open(image_bytes)
        
        # Preprocess the image for better text extraction
        image = preprocess_image(image)
        
        # Convert the image to a numpy array
        image_np = np.array(image)
        
        # Use EasyOCR to extract text from the image
        results = reader.readtext(image_np)
        
        if not results:
            raise ValueError("No text found in the image.")
        
        # Concatenate all the extracted text
        return ' '.join([result[1] for result in results])
    except Exception as e:
        raise RuntimeError(f"Error extracting text from image: {str(e)}")

def parse_soil_test_data(text):
    patterns = {
        'nitrogen': r'Nitrogen\s*Level\s*(\d+(?:\.\d+)?)\s*ppm',
        'phosphorus': r'Phosphorus\s*(\d+(?:\.\d+)?)\s*ppm',
        'potassium': r'Potassium\s*(\d+(?:\.\d+)?)\s*ppm',
        'ph': r'pH\s*level\s*(\d+(?:\.\d+)?)'
    }
    
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        results[key] = float(match.group(1)) if match else 'N/A'
    
    return results

def ppm_to_lbs_per_acre(ppm):
    return ppm * 2 if ppm != 'N/A' else 'N/A'

@app.route('/ocrScanner', methods=['GET', 'POST'])
def ocr_scanner():
    results = {}
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                text = extract_text_from_image(file)
                soil_data = parse_soil_test_data(text)
                results = {
                    'nitrogen': ppm_to_lbs_per_acre(soil_data.get('nitrogen')),
                    'phosphorus': ppm_to_lbs_per_acre(soil_data.get('phosphorus')),
                    'potassium': ppm_to_lbs_per_acre(soil_data.get('potassium')),
                    'ph': soil_data.get('ph') if soil_data.get('ph') is not None else 'N/A'
                }
            except (ValueError, RuntimeError) as e:
                flash(f'Error processing file: {str(e)}')
        else:
            flash('Invalid file type or no file selected')
    
    return render_template('index.html', results=results)

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
