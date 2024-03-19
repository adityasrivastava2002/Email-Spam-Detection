import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from speech_recognition import Microphone, Recognizer
from transformers import pipeline
from googletrans import Translator

# Load the Multinomial Naive Bayes classifier and CountVectorizer
classifier = pickle.load(open('multinomial_nb_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

translator = Translator()

def transform_text(text):
    # Implement text transformation logic 
    return text

def analyze_sentiment(text):
    sentiment_pipeline = pipeline('sentiment-analysis')
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

def calculate_certainty(predictions):
    sorted_probs = sorted(predictions, reverse=True)
    certainty = sorted_probs[0] - sorted_probs[1]
    return certainty

def plot_confidence_accuracy(confidence, accuracy):
    labels = ["Model's Accuracy", "Sentiment's Accuracy"]
    values = [confidence, accuracy]

    fig, ax = plt.subplots(figsize=(3, 2))
    bars = ax.bar(labels, values, color=['blue', 'green'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom', fontsize=5)

    plt.title('Sentimental and Model Accuracy', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    st.pyplot(fig)

def speech_to_text():
    recognizer = Recognizer()
    with Microphone() as source:
        st.write("Say something...")
        audio_data = recognizer.listen(source, timeout=5)

    try:
        st.write("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Initialize session state
if 'recorded_once' not in st.session_state:
    st.session_state.recorded_once = False
    st.session_state.input_sms = None

# st.title("Email/SMS Spam Classifier")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Email/SMS Spam Classifier</h2>
</div><br><br>
"""
st.markdown(html_temp,unsafe_allow_html=True)

with st.sidebar:
    input_type = st.radio("Choose input type:", ["Text", "Speech"])

if input_type == "Text":
    st.session_state.input_sms = translator.translate(st.text_area("Enter the message in any language"), dest = "en").text
    st.session_state.recorded_once = False
elif not st.session_state.recorded_once and input_type == "Speech":
    if st.button("Start recording"):
        st.session_state.input_sms = speech_to_text()
        st.session_state.recorded_once = True
    if st.session_state.input_sms:
        st.text_area("Transcribed Text", st.session_state.input_sms)

if st.button('Predict') and st.session_state.input_sms:
    # 1. preprocess
    transformed_sms = transform_text(st.session_state.input_sms)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_sms])
    # 3. predict
    category_prediction = classifier.predict_proba(vector_input)[0]
    # 4. sentiment analysis
    sentiment_label, sentiment_score = analyze_sentiment(transformed_sms)
    # 5. Display
    st.write("Input: "+st.session_state.input_sms)
    if category_prediction[1]>0.5:
        st.header("Result: Spam")
    else:
        st.header("Result: Not Spam")
    st.write("Model Score: {:.4f}".format(calculate_certainty(category_prediction)))
    st.header("Sentiment Analysis: "+sentiment_label.capitalize())
    st.write("Sentiment Score: {:.4f}".format(sentiment_score))

    # Plotting confidence and accuracy
    plot_confidence_accuracy(calculate_certainty(category_prediction), sentiment_score)  # Replace 0.8 with your actual accuracy

    st.session_state.recorded_once = False
    st.session_state.input_sms = None
