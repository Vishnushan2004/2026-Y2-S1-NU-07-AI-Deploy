import re
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 150

@st.cache_resource
def load():
    model = tf.keras.models.load_model("lstm_sentiment.keras", compile=False)
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())
    return model, tok

model, tokenizer = load()

def clean_text(t):
    t = re.sub(r"[^a-z0-9\s]", "", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

st.title("Movie Review Sentiment Analysis")
review = st.text_area("Movie Review", height=150,
                      placeholder="Enter a movie review here...")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        seq = tokenizer.texts_to_sequences([clean_text(review)])
        pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
        p = float(model.predict(pad, verbose=0)[0][0])
        label = "Positive" if p >= 0.5 else "Negative"
        st.subheader(label)
        st.metric("Model confidence", f"{max(p, 1 - p):.4f}")
