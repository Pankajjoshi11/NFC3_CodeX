from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

def predict_yield(model_path, input_values):
    model = joblib.load(model_path)
    input_df = pd.DataFrame([input_values])
    predicted_yield = model.predict(input_df)
    return predicted_yield[0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_values = {
            'Rain Fall (mm)': float(request.form['rain_fall']),
            'Fertilizer': float(request.form['fertilizer']),
            'Temperature': float(request.form['temperature']),
            'Nitrogen (N)': float(request.form['nitrogen']),
            'Phosphorus (P)': float(request.form['phosphorus']),
            'Potassium (K)': float(request.form['potassium'])
        }
        model_path = 'models/crop_yield_model.pkl'
        predicted_yield = predict_yield(model_path, input_values)
        return render_template('index.html', predicted_yield=predicted_yield)
    
    return render_template('index.html', predicted_yield=None)

if __name__ == '__main__':
    app.run(debug=True,port=5003)
