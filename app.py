import os
import pandas as pd
import numpy as np
import joblib
import plotly
import plotly.express as px
import plotly.graph_objs as go
import json
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

def load_dataset():
    try:
        df = pd.read_csv("dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
        print(f"Loaded dataset with {len(df)} records and {df['Station_Name'].nunique()} stations")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_model_and_scaler():
    try:
        # Load the model
        model_path = os.path.join("model", "linear_regression_model.pkl")
        model = joblib.load(model_path)
        print("Model loaded successfully")
        # Load the standard scaler
        scaler_path = os.path.join("model", "linear_regression_standard_scaler.pkl")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully")
        
        # Define the expected feature names (must match the order and names used during scaler training)
        feature_names = [
            "Air Temperature (C°)",
            "Air Temperature Uncertainty (C°)",
            "Wind Direction at 3m (°N)",
            "Wind Direction at 3m Uncertainty (°N)",
            "Wind Speed at 3m (m/s)",
            "Wind Speed at 3m Uncertainty (m/s)",
            "Wind Speed at 3m (std dev) (m/s)",
            "DHI (Wh/m2)",
            "DHI Uncertainty (Wh/m2)",
            "Standard Deviation DHI (Wh/m2)",
            "DNI (Wh/m2)",
            "DNI Uncertainty (Wh/m2)",
            "Standard Deviation DNI (Wh/m2)",
            "GHI Uncertainty (Wh/m2)",
            "Standard Deviation GHI (Wh/m2)",
            "Peak Wind Speed at 3m (m/s)",
            "Peak Wind Speed at 3m Uncertainty (m/s)",
            "Relative Humidity (%)",
            "Relative Humidity Uncertainty (%)",
            "Barometric Pressure (mB (hPa equiv))",
            "Barometric Pressure Uncertainty (mB (hPa equiv))"
        ]
        
        # Store feature names with the model and scaler objects
        model.feature_names_in_ = feature_names
        scaler.feature_names_in_ = feature_names  # Ensure scaler knows the feature names
        print(f"Model expects {len(model.feature_names_in_)} features")
        print(f"Model feature names: {model.feature_names_in_}")
        
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

dataset = load_dataset()
model, scaler = load_model_and_scaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map-data')
def map_data():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500
    
    unique_stations = dataset.drop_duplicates(subset=['Station_Name'])
    avg_ghi = dataset.groupby('Station_Name')['GHI (Wh/m2)'].mean().to_dict()
    
    map_data = []
    for _, row in unique_stations.iterrows():
        station_data = {
            "station_name": row['Station_Name'],
            "latitude": float(row['Latitude']),
            "longitude": float(row['Longitude']),
            "avg_ghi": float(avg_ghi[row['Station_Name']])
        }
        map_data.append(station_data)
    
    return jsonify(map_data)

@app.route('/get-station-names')
def get_station_names():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500
    
    return jsonify({"stations": dataset['Station_Name'].unique().tolist()})

@app.route('/station-details')
def station_details():
    station_name = request.args.get('station', '')
    year_filter = request.args.get('year', '')
    
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500
    
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Latitude', 'Longitude']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    station_data_full = dataset[dataset['Station_Name'] == station_name]
    
    if year_filter and year_filter.isdigit():
        year = int(year_filter)
        station_data_filtered = station_data_full[station_data_full['Date'].dt.year == year]
        if station_data_filtered.empty:
            station_data_filtered = station_data_full
    else:
        station_data_filtered = station_data_full
    
    if station_data_filtered.empty:
        return jsonify({"error": "No data found for this station"}), 404
    
    first_row = station_data_filtered.iloc[0]
    details = {
        "station_name": station_name,
        "longitude": float(first_row['Longitude']),
        "latitude": float(first_row['Latitude']),
        "date": first_row['Date'].strftime('%Y-%m-%d %H:%M')
    }
    
    monthly_data = station_data_filtered.groupby(station_data_filtered['Date'].dt.month)[numerical_cols].mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    all_months = pd.DataFrame(index=range(1, 13))
    monthly_data = all_months.join(monthly_data)
    monthly_data = monthly_data.fillna(0)
    
    monthly_chart_data = {
        "months": months,
        "data": {col: monthly_data[col].tolist() for col in numerical_cols}
    }
    
    mean_values = station_data_filtered[numerical_cols].mean().to_dict()
    for key in mean_values:
        mean_values[key] = float(mean_values[key])
    
    return jsonify({
        "details": details,
        "monthly_chart_data": monthly_chart_data,
        "mean_values": mean_values
    })

@app.route('/data-analysis')
def data_analysis():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500
    
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Latitude', 'Longitude']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    summary_stats = {}
    for col in numerical_cols:
        for stat in ['mean', 'min', 'max', 'std']:
            stat_key = f"{col}_{stat}"
            summary_stats[stat_key] = dataset.groupby('Station_Name')[col].agg(stat).to_dict()
    
    return jsonify({"summary_stats": summary_stats})

@app.route('/station-comparison', methods=['POST'])
def station_comparison():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500
    
    data = request.json
    stations = data.get('stations', [])
    params = data.get('params', [])
    year = data.get('year')
    
    if not stations or not params:
        return jsonify({"error": "No stations or parameters selected"}), 400
    
    filtered_data = dataset[dataset['Station_Name'].isin(stations)]
    
    if year and year.isdigit():
        year_int = int(year)
        filtered_data = filtered_data[filtered_data['Date'].dt.year == year_int]
    
    if filtered_data.empty:
        return jsonify({"error": "No data found for selected stations"}), 404
    
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Latitude', 'Longitude']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    dates = {}
    values = {}
    
    for station in stations:
        station_data = filtered_data[filtered_data['Station_Name'] == station]
        if not station_data.empty:
            dates[station] = station_data['Date'].dt.strftime('%Y-%m-%d').tolist()
            values[station] = {}
            for param in numerical_cols:
                if param in station_data.columns:
                    values[station][param] = station_data[param].tolist()
                else:
                    values[station][param] = []
    
    summary_stats = {}
    for param in numerical_cols:
        for stat in ['mean', 'min', 'max', 'std']:
            key = f"{param}_{stat}"
            summary_stats[key] = {}
            for station in stations:
                station_data = filtered_data[filtered_data['Station_Name'] == station]
                if not station_data.empty and param in station_data.columns:
                    value = station_data[param].agg(stat)
                    summary_stats[key][station] = round(float(value), 2) if pd.notnull(value) else None
                else:
                    summary_stats[key][station] = None
    
    return jsonify({
        "dates": dates,
        "values": values,
        "summary_stats": summary_stats
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        input_data = request.json
        print(f"Raw input: {input_data}")  # Debugging
        
        # Get the feature names stored with the model
        required_features = model.feature_names_in_
        
        # Validate inputs
        for feature in required_features:
            if feature not in input_data or input_data[feature] is None or pd.isna(input_data[feature]):
                return jsonify({"error": f"Missing or invalid feature: {feature}"}), 400

        # Create input DataFrame with correct feature order
        prediction_df = pd.DataFrame([input_data], columns=required_features)

        # Ensure all required features are populated
        missing_features = [f for f in required_features if prediction_df[f].isnull().any()]
        if missing_features:
            return jsonify({"error": f"Missing values for features: {missing_features}"}), 400

        # Debugging: Print the DataFrame before scaling
        print(f"Prediction DataFrame: {prediction_df}")

        # Scale the input features
        scaled_features = scaler.transform(prediction_df[required_features])
        
        # Debugging: Print scaled features
        print(f"Scaled features: {scaled_features}")

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # If prediction is a multi-dimensional array (common with Keras models),
        # extract the first value
        if hasattr(prediction, '__len__') and len(prediction) > 0:
            prediction = prediction[0]
            
        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/get-ghi-thresholds')
def get_ghi_thresholds():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500
    
    try:
        ghi = dataset['GHI (Wh/m2)']
        return jsonify({
            'low': float(np.percentile(ghi, 25)),
            'high': float(np.percentile(ghi, 75))
        })
    except Exception as e:
        print(f"Threshold calculation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-average-values')
def get_average_values():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500

    # Use the feature names stored with the model
    required_features = model.feature_names_in_

    averages = {}
    for feature in required_features:
        if feature in dataset.columns:
            averages[feature] = float(dataset[feature].mean())
        else:
            averages[feature] = None  # Handle missing columns

    return jsonify(averages)

if __name__ == '__main__':
    app.run(debug=True)