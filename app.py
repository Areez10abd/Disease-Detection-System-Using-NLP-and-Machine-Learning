from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the model and vectorizer
tfidf_vectorizer = joblib.load('tfidfvectorizer.pk1')
model = joblib.load('passmodel.pk1')

# Preprocessing function
def review_to_words(raw_review):
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

    # 2. Remove non-letters
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)

    # 3. Convert to lowercase and split
    words = letters_only.lower().split()

    # 4. Remove stopwords
    meaningful_words = [w for w in words if not w in stop]

    # 5. Lemmatize
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]

    # 6. Join words back into one string separated by space
    return ' '.join(lemmatized_words)

# Dictionary for disease advice and medicines
disease_info = {
    "Bacterial Infection": {
        "medicines": ["Amoxicillin", "Ciprofloxacin", "Doxycycline"],
        "advice": "Stay hydrated, follow the full course of antibiotics, and consult a doctor if symptoms persist."
    },
    "Depression": {
        "medicines": ["Fluoxetine", "Sertraline", "Citalopram"],
        "advice": "Seek therapy, consider medication under a doctor's supervision, and maintain a support network."
    },
    "Diabetes, Type 2": {
        "medicines": ["Metformin", "Glimepiride", "Pioglitazone"],
        "advice": "Monitor blood sugar levels, maintain a healthy diet, and exercise regularly."
    },
    "High Blood Pressure": {
        "medicines": ["Amlodipine", "Lisinopril", "Losartan"],
        "advice": "Reduce sodium intake, exercise regularly, and consult your doctor for medication adjustments."
    },
    "Migraine": {
        "medicines": ["Sumatriptan", "Rizatriptan", "Propranolol"],
        "advice": "Avoid known triggers, stay hydrated, and consider preventive medications if migraines are frequent."
    }
}

@app.route('/')
def index():
    return render_template('disease_detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Preprocess the text
    clean_text = review_to_words(text)

    # Transform the input using the vectorizer
    tfidf_text = tfidf_vectorizer.transform([clean_text])

    # Predict using the trained model
    prediction = model.predict(tfidf_text)[0]

    # Fetch disease-specific advice and medicines
    top_medicines = disease_info[prediction]["medicines"]
    advice = disease_info[prediction]["advice"]

    return jsonify(prediction=prediction, medicines=top_medicines, advice=advice)

if __name__ == '__main__':
    app.run(debug=True)
