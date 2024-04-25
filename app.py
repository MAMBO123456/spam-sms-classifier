import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Imp resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    filtered_tokens = []
    for token in tokens:
        if token.isalnum() and token not in stopwords.words('english') and token not in string.punctuation:
            filtered_tokens.append(ps.stem(token))

    return " ".join(filtered_tokens)

# Load TF-IDF Vectorizer and Model
with open('vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess text
        transformed_sms = transform_text(input_sms)
        # Vectorize text
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display prediction
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
