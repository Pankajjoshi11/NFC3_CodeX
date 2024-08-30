from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

def load_model(model_path):
    """Load the trained model from a file."""
    return joblib.load(model_path)

def prepare_data(input_data, feature_names):
    """Prepare input data for prediction."""
    # Convert input_data to DataFrame
    df = pd.DataFrame([input_data], columns=feature_names)
    
    # Handle missing values and standardize features
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    df = imputer.fit_transform(df)
    df = scaler.fit_transform(df)
    
    return df

def predict_soil_type(model, data):
    """Predict the soil type based on the input data."""
    return model.predict(data)

@app.route('/', methods=['GET', 'POST'])
def home():
    model_path = 'models/soil_classifier_model.pkl'
    model = load_model(model_path)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    predicted_soil_type = None
    
    if request.method == 'POST':
        # Get form data
        input_data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Prepare data and predict
        prepared_data = prepare_data(input_data, feature_names)
        prediction = predict_soil_type(model, prepared_data)
        predicted_soil_type = prediction[0]
    
    return render_template('index.html', predicted_soil_type=predicted_soil_type)

if __name__ == '__main__':
    app.run(debug=True,port=5004)
