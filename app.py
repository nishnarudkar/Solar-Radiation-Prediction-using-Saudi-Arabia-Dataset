import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from src.dataset import clean_column_names, FEATURE_NAMES
from src.predict import SolarPredictor

app = Flask(__name__)
CORS(app)

predictor = SolarPredictor()


def load_dataset():
    try:
        df = pd.read_csv("dataset.csv")
        df = clean_column_names(df)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
        print(f"Loaded dataset with {len(df)} records and {df['Station_Name'].nunique()} stations")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def load_model_and_scaler():
    try:
        model_path = os.path.join("model", "linear_regression_model.pkl")
        scaler_path = os.path.join("model", "linear_regression_standard_scaler.pkl")
        model = joblib.load(model_path) if os.path.exists(model_path) else None
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        return model, scaler
    except Exception as e:
        print(f"Error loading baseline model or scaler: {e}")
        return None, None


dataset = load_dataset()
model, scaler = load_model_and_scaler()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models')
def get_models():
    """Return list of available machine learning and deep learning models."""
    return jsonify({"models": predictor.get_available_models()})


@app.route('/map-data')
def map_data():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500

    unique_stations = dataset.drop_duplicates(subset=['Station_Name'])
    avg_ghi = dataset.groupby('Station_Name')['GHI (Wh/m2)'].mean().to_dict()

    map_data_list = []
    for _, row in unique_stations.iterrows():
        station_data = {
            "station_name": row['Station_Name'],
            "latitude": float(row['Latitude']),
            "longitude": float(row['Longitude']),
            "avg_ghi": float(avg_ghi[row['Station_Name']])
        }
        map_data_list.append(station_data)

    return jsonify(map_data_list)


@app.route('/stations')
def stations():
    if dataset is None:
        return jsonify({"error": "No dataset loaded"}), 500

    return jsonify({"stations": dataset['Station_Name'].unique().tolist()})


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

    numerical_cols = [c for c in dataset.select_dtypes(include=[np.number]).columns if c not in ['Latitude', 'Longitude', 'Year', 'Month']]

    station_data_full = dataset[dataset['Station_Name'] == station_name]

    if year_filter and year_filter.isdigit():
        year = int(year_filter)
        station_data_filtered = station_data_full[station_data_full['Year'] == year]
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

    monthly_data = station_data_filtered.groupby('Month')[numerical_cols].mean()
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

    numerical_cols = [c for c in dataset.select_dtypes(include=[np.number]).columns if c not in ['Latitude', 'Longitude', 'Year', 'Month']]

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

    data = request.json or {}
    stations = data.get('stations', [])
    params = data.get('params', [])
    year = data.get('year')

    if not stations or not params:
        return jsonify({"error": "No stations or parameters selected"}), 400

    filtered_data = dataset[dataset['Station_Name'].isin(stations)]

    if year and str(year).isdigit():
        year_int = int(year)
        filtered_data = filtered_data[filtered_data['Year'] == year_int]

    if filtered_data.empty:
        return jsonify({"error": "No data found for selected stations"}), 404

    numerical_cols = [col for col in dataset.select_dtypes(include=[np.number]).columns if col not in ['Latitude', 'Longitude', 'Year', 'Month']]

    dates = {}
    values = {}

    for station in stations:
        st_df = filtered_data[filtered_data['Station_Name'] == station]
        if not st_df.empty:
            dates[station] = st_df['Date_str'].tolist()
            values[station] = {
                col: st_df[col].tolist()
                for col in numerical_cols if col in st_df.columns
            }
        else:
            dates[station] = []
            values[station] = {}

    # Fast vectorized summary statistics calculation
    agg_df = filtered_data.groupby('Station_Name')[numerical_cols].agg(['mean', 'min', 'max', 'std'])
    summary_stats = {}
    for col in numerical_cols:
        for stat in ['mean', 'min', 'max', 'std']:
            key = f"{col}_{stat}"
            if (col, stat) in agg_df.columns:
                series = agg_df[(col, stat)]
                summary_stats[key] = {
                    st: round(float(series[st]), 2) if (st in series.index and pd.notnull(series[st])) else None
                    for st in stations
                }
            else:
                summary_stats[key] = {st: None for st in stations}

    return jsonify({
        "dates": dates,
        "values": values,
        "summary_stats": summary_stats
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json or {}
        default_model = "production" if os.path.exists(os.path.join("model", "production_dl_model.pt")) else "linear_regression"
        model_type = input_data.get("model_type") or default_model

        # Run inference using unified predictor
        prediction_val = predictor.predict(input_data, model_type=model_type)

        return jsonify({
            "prediction": round(float(prediction_val), 2),
            "model_used": model_type,
            "unit": "Wh/m2"
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
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

    averages = {}
    for feature in FEATURE_NAMES:
        if feature in dataset.columns:
            averages[feature] = float(dataset[feature].mean())
        else:
            averages[feature] = None

    return jsonify(averages)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)