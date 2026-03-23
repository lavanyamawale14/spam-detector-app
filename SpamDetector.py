import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page settings
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Dark theme styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .title {
        text-align:center;
        font-size:42px;
        font-weight:bold;
        color:#ff914d;
        margin-bottom:10px;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        color:#cfcfcf;
        margin-bottom:30px;
    }
    .stTextArea textarea {
        background-color: #262730;
        color: white;
        border-radius: 12px;
        padding: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff914d);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        transition: 0.2s;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📧 Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message is Spam or Not Spam using AI</div>', unsafe_allow_html=True)

# Load data
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']
model = MultinomialNB()
model.fit(X, y)

# Input container
with st.container():
    st.subheader("✉ Enter your message")
    msg = st.text_area("", height=150)

# Prediction
if st.button("🔍 Check Message"):
    if msg.strip() == "":
        st.warning("⚠ Please enter a message")
    else:
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)
        prob = model.predict_proba(vec)

        spam_prob = prob[0][1] * 100
        ham_prob = prob[0][0] * 100

        st.write("")
        if pred[0] == 1:
            st.error(f"🚫 Spam Message\n\nSpam Probability: {spam_prob:.2f}%")
            st.progress(int(spam_prob))
        else:
            st.success(f"✅ Not a Spam Message\n\nSafe Probability: {ham_prob:.2f}%")
            st.progress(int(ham_prob))