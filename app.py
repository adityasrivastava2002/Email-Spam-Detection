import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from speech_recognition import Microphone, Recognizer
from textblob import TextBlob

# Load the Multinomial Naive Bayes classifier and CountVectorizer
classifier = pickle.load(open('multinomial_nb_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

def transform_text(text):
    # Implement your own text transformation logic if needed
    return text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Classify the sentiment as 'positive', 'neutral', or 'negative'
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

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

st.title("Email/SMS Spam Classifier")

input_type = st.radio("Choose input type:", ["Text", "Speech"])



if input_type == "Text":
    st.session_state.input_sms = st.text_area("Enter the message")
    st.session_state.recorded_once = False  # Reset recording state
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
    category_prediction = classifier.predict(vector_input)[0]
    # 4. sentiment analysis
    sentiment = analyze_sentiment(transformed_sms)
    # 5. Display
    st.write("Input: "+st.session_state.input_sms)
    if category_prediction=='spam':
        st.header("Result: Spam")
    else:
        st.header("Result: Not Spam")
    st.header("Sentiment Analysis: "+sentiment.capitalize())
    # st.write(f"The sentiment of the message is {sentiment.capitalize()}")
    st.session_state.recorded_once = False
    st.session_state.input_sms = None
