import re
import gradio as gr
import tensorflow as tf
import spaces

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Configuration
# -----------------------------
MAX_LENGTH = 150
MODEL_PATH = "lstm_sentiment.keras"
TOKENIZER_PATH = "tokenizer.json"


# -----------------------------
# Load trained model and tokenizer
# -----------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())


# -----------------------------
# Clean movie review
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Predict sentiment
# -----------------------------
@spaces.GPU(duration=60)
def predict_sentiment(review):
    if not review or not review.strip():
        return "Please enter a movie review.", 0.0

    cleaned = clean_text(review)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)
    probability = float(prediction[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1.0 - probability

    return sentiment, round(confidence, 4)


# -----------------------------
# Gradio interface
# -----------------------------
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        label="Movie Review",
        placeholder="Enter a movie review here..."
    ),
    outputs=[
        gr.Textbox(label="Predicted Sentiment"),
        gr.Number(label="Model Confidence")
    ],
    title="Movie Review Sentiment Analysis",
    description=(
        "Enter a movie review to predict whether its sentiment "
        "is positive or negative using our trained LSTM model. "
        "The confidence value is the model's output-based score, "
        "not a guarantee of prediction correctness."
    ),
    examples=[
        ["This movie was fantastic and the acting was brilliant."],
        ["A boring movie with terrible acting and a weak story."]
    ],
    flagging_mode="never"
)


# -----------------------------
# Launch application
# -----------------------------
if __name__ == "__main__":
    demo.launch()
