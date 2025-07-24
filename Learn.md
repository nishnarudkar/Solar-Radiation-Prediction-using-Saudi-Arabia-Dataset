# Learn: Solar Radiation Prediction using Saudi Arabia Dataset

Welcome to the learning guide for the **Solar Radiation Prediction using Saudi Arabia Dataset** project!  
This document will help you understand the key concepts, technologies, and steps involved in the repository so you can contribute, extend, or use the project effectively.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Concepts](#key-concepts)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Getting Started](#getting-started)
6. [Model Training & Evaluation](#model-training--evaluation)
7. [Web Application Deployment](#web-application-deployment)
8. [Useful Resources](#useful-resources)

---

## Project Overview

This project predicts **Global Horizontal Irradiance (GHI)**—the solar radiation received per unit area on a horizontal surface—using machine learning models trained on historical weather data from Saudi Arabia.  
The aim is to provide accurate, real-time GHI forecasting via a web application.

---

## Key Concepts

- **Global Horizontal Irradiance (GHI):** Key metric for solar energy potential, measured in W/m².
- **Machine Learning Models:** Algorithms like Linear Regression, Random Forest, or Neural Networks to predict GHI.
- **Feature Engineering:** Selecting and preparing weather variables (temperature, humidity, wind speed, etc.) as inputs for models.
- **Web Application:** User-friendly interface for real-time predictions.

---

## Project Structure

```
.
├── data/                # Raw and processed datasets
├── models/              # Trained machine learning models
├── notebooks/           # Jupyter notebooks for EDA, training, and evaluation
├── static/              # Static files for web app (JS, CSS)
├── templates/           # HTML templates for web app
├── app.py / server.js   # Web application backend (Flask or Node.js/Express)
├── requirements.txt     # Python dependencies
├── README.md            # Project overview
├── learn.md             # (This file)
```

---

## How It Works

1. **Data Collection:** Gather historical weather and solar radiation data for Saudi Arabia.
2. **Exploratory Data Analysis (EDA):** Analyze data using Python notebooks to identify trends and correlations.
3. **Feature Engineering:** Select relevant features and preprocess the data (cleaning, normalization).
4. **Model Training:** Train ML models to predict GHI.
5. **Evaluation:** Measure model performance using metrics like RMSE, MAE, etc.
6. **Deployment:** Integrate the best model into a web app for real-time predictions.

---

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nishnarudkar/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset.git
   cd Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset
   ```

2. **Install Dependencies**
   - For Python:
     ```bash
     pip install -r requirements.txt
     ```
   - For Node.js (if using JS backend):
     ```bash
     npm install
     ```

3. **Explore Notebooks**
   - Open the `notebooks/` directory to review EDA, preprocessing, and model training steps.

4. **Run the Web Application**
   - For Python (Flask):
     ```bash
     python app.py
     ```
   - For Node.js/Express:
     ```bash
     node server.js
     ```

---

## Model Training & Evaluation

- Review and modify the Jupyter notebooks for:
  - Data cleaning and preprocessing
  - Feature selection and engineering
  - Model selection, training, and tuning
  - Validation and performance metrics

- Trained models are saved in the `models/` directory for deployment.

---

## Web Application Deployment

- The web app uses HTML, JS, and either Flask (Python) or Express (JS) as the backend.
- Users can input weather data to get instant GHI predictions.
- The interface and logic can be extended for more features or different datasets.

---

## Useful Resources

- [Solar Radiation Data Basics (NREL)](https://www.nrel.gov/grid/solar-resource.html)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Node.js Documentation](https://nodejs.org/en/docs/)
- [Saudi Arabia Meteorological Data](https://www.ncdc.noaa.gov/)

---

## Contributing

See `README.md` for contributing guidelines.

---

Happy Learning!  
For questions, open an issue or contact the repository maintainer.
