import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
cv = joblib.load('countvectorizer.joblib')

# NLTK setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Preprocessing
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = ''.join([c if c.isalnum() else ' ' for c in text])
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Page config
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide")

# Background and header
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #ffecd2, #fcb69f);
    }
    .review-box {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.2);
    }
    .button-style {
        background: #ff7e5f;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; color:#FF4500;'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Check if a review is Positive or Negative!</p>", unsafe_allow_html=True)

# Input container
st.markdown("<div class='review-box'>", unsafe_allow_html=True)
review_input = st.text_area("✍️ Write your review here:", height=150)
st.markdown("</div>", unsafe_allow_html=True)

# Predict button
if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        processed_review = clean_text(review_input)
        vector_input = cv.transform([processed_review])
        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.markdown(
                "<div style='text-align:center; padding:25px; background-color:#d4edda; border-radius:15px; box-shadow: 2px 2px 12px #aaa;'>"
                "<h2 style='color:#155724; font-size:28px;'>😊 Positive Review!</h2></div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='text-align:center; padding:25px; background-color:#f8d7da; border-radius:15px; box-shadow: 2px 2px 12px #aaa;'>"
                "<h2 style='color:#721c24; font-size:28px;'>😞 Negative Review!</h2></div>", unsafe_allow_html=True
            )

# Footer
st.markdown(
    "<div style='text-align:center; padding:10px; color:#555;'>Powered by Python & Streamlit 🎬</div>",
    unsafe_allow_html=True
)



