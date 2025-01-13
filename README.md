Disease Detection System Using NLP and Machine Learning 🩺🤖
Overview
This project aims to predict diseases based on patient symptoms using Natural Language Processing (NLP) and machine learning. It combines text preprocessing, machine learning, and a Flask-based web app to deliver accurate disease predictions and health recommendations.

🛠️ Key Features
Dataset: Sourced from Kaggle, focusing on diseases like Bacterial Infection, Depression, Diabetes Type 2, High Blood Pressure, and Migraines.
Data Visualization: Word clouds to highlight common symptoms for each condition.
Text Preprocessing: Cleaning text by removing HTML, non-alphabetic characters, and stopwords, with lemmatization for better analysis.
Feature Extraction: Vectorized text using TF-IDF for machine learning.
ML Model: Trained a Passive Aggressive Classifier for high-accuracy predictions.
Model Deployment: Saved the trained model and vectorizer for use in a Flask-based app.
🚀 How It Works
1️⃣ Input: Patients enter their symptoms in the web app.
2️⃣ Processing: The app cleans and processes the input text using NLP techniques.
3️⃣ Prediction: The trained ML model predicts the disease.
4️⃣ Recommendations: The app suggests medicines and provides doctor’s advice for better health management.

💻 Tech Stack
Python: scikit-learn, pandas, numpy, nltk
Web Framework: Flask
Visualization: Word clouds, Confusion Matrix
This project showcases the power of NLP and machine learning in improving healthcare.
