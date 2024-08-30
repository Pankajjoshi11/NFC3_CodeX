import pandas as pd
import joblib

def predict_yield(model_path, input_values):
    model = joblib.load(model_path)
    input_df = pd.DataFrame([input_values])
    predicted_yield = model.predict(input_df)
    return predicted_yield[0]

if __name__ == "__main__":
    input_values = {
        'Rain Fall (mm)': 1230.0,
        'Fertilizer': 80.0,
        'Temperatue': 28,
        'Nitrogen (N)': 80.0,
        'Phosphorus (P)': 24.0,
        'Potassium (K)': 20.0
    }
    predicted_yield = predict_yield('models/crop_yield_model.pkl', input_values)
    print(f'Predicted Yield (Q/acre): {predicted_yield}')
