# Neuro_Tracker
Tremor Detection
Neuro Tremor Detection Web App

Description:
A web-based application to detect Parkinson’s tremors (rest, postural, and kinetic) from sensor data (gyroscope & accelerometer) using a trained machine learning model. Users can upload CSV files or enter manual sensor readings for real-time predictions.

Features:
Predict tremor severity: No Tremor, Mild, Moderate, Severe.
Accepts CSV files with sensor features or manual input.
Shows predicted label and confidence score.
Compatible with Gradio for interactive web UI.

Requirements:
Python 3.10+

Libraries: pandas, numpy, scikit-learn, joblib, gradio


Notes:

CSV must contain the same features used during model training.
Missing features will cause prediction errors.
