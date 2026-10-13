
import re
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 150

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "lstm_sentiment.keras",
        compile=False
    )

    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    return model, tokenizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review to predict whether its sentiment "
    "is positive or negative using our trained LSTM model."
)

try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error("The model could not be loaded. Check the deployment logs.")
    st.exception(e)
    st.stop()

review = st.text_area(
    "Enter your movie review",
    placeholder="Example: This movie was fantastic!",
    height=150
)

if st.button("Predict Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        cleaned = clean_text(review)
        sequence = tokenizer.texts_to_sequences([cleaned])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        probability = float(
            model.predict(padded, verbose=0)[0][0]
        )

        if probability >= 0.5:
            sentiment = "Positive"
            score = probability
        else:
            sentiment = "Negative"
            score = 1 - probability

        if sentiment == "Positive":
            st.success(f"Predicted sentiment: {sentiment}")
        else:
            st.error(f"Predicted sentiment: {sentiment}")

        st.metric("Model confidence-like score", f"{score:.2%}")
        st.caption(
            "This score is derived from the model output; "
            "it is not a guarantee that the prediction is correct."
        )
