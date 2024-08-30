import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

def main():
    # Path to the trained model file
    model_path = 'path/to/your/soil_classifier_model.pkl'
    
    # Load the trained model
    model = load_model(model_path)
    
    # Define feature names (should match those used during training)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Example input data (update with actual values as needed)
    input_data = {
        'N': 50,
        'P': 30,
        'K': 40,
        'temperature': 20,
        'humidity': 60,
        'ph': 6.5,
        'rainfall': 120
    }
    
    # Prepare data for prediction
    prepared_data = prepare_data(input_data, feature_names)
    
    # Predict soil type
    prediction = predict_soil_type(model, prepared_data)
    
    # Display results
    print(f'Predicted Soil Type: {prediction[0]}')

if __name__ == '__main__':
    main()
