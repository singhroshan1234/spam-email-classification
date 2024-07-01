import streamlit as st
import pickle
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and feature extraction
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as feature_file:
    feature_extraction = pickle.load(feature_file)

# Initialize stemmer
ps = PorterStemmer()

# Streamlit UI
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")


# Function to add background image
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1555617991-286575d18082");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


add_bg_from_url()

st.title("📧 Spam Email Classifier")
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    .stTextInput {font-size: 18px; height: 200px;}
    .stButton button {background-color: #4CAF50; color: red; font-size: 20px; padding: 10px 24px; border: none; border-radius: 12px;}
    .stButton button:hover {background-color: #45a049;}
    </style>
    """,
    unsafe_allow_html=True
)

st.write(
    "This application classifies whether an email is **spam** or **ham** (not spam). Enter the email text below and click **Classify** to see the result.")

input_email = st.text_area("Input Email", height=200)


# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Classify"):
    if input_email.strip() != "":
        transformed_email = transform_text(input_email)
        input_data_features = feature_extraction.transform([transformed_email])
        prediction = model.predict(input_data_features)

        if prediction[0] == 1:
            st.error("Prediction: spam Email")
        else:
            st.success("Prediction: ham email")
    else:
        st.warning("Please enter some text in the email input field.")


