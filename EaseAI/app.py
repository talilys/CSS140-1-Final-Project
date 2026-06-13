"""
app.py
EaseAI - Student Stress and Mood Detection Chatbot
Improved: confidence-based unclear detection, richer responses,
context-aware follow-ups, and better crisis handling.
"""

import re
import pickle
import random
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Confidence threshold below which we ask for more context
UNCLEAR_THRESHOLD = 0.50

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------
# Response Bank — varied, empathetic, student-focused
# -----------------------------------------------------------
RESPONSES = {
    "positive": [
        "That's really great to hear! It sounds like things are going well — hold onto that feeling. You've earned it.",
        "I love hearing this! Moments like these are worth celebrating, even the small ones. Keep going.",
        "That's wonderful. Positive progress, no matter the size, really adds up. You're doing better than you think.",
        "So glad things are looking up! It sounds like your efforts are paying off. Keep that momentum.",
        "That made me smile reading it. You deserve to feel good about where you are right now.",
        "It's genuinely nice to hear you're doing well. Remember this feeling when things get tough — it's proof you can get here again.",
        "Yes! That's the kind of win worth pausing to appreciate. Well done.",
    ],
    "negative": [
        "That sounds really hard, and I just want you to know — what you're feeling makes complete sense. You're carrying a lot right now.",
        "It's okay to feel overwhelmed. School can pile up fast. Try to take it one task at a time, even just the next smallest thing.",
        "I hear you. Stress like this is real and it's heavy. Be gentle with yourself — you're doing the best you can in a tough moment.",
        "You're not alone in feeling this way. A lot of students go through this. If you can, take a short break — even five minutes of breathing can reset things a little.",
        "That sounds exhausting. When everything feels like too much, it sometimes helps to write it all down and just pick one thing to start with.",
        "Feeling this way doesn't mean you're failing — it means you care. But you also need rest. Please don't forget to take care of yourself too.",
        "I'm really sorry you're going through this. If things feel unmanageable, please consider talking to a school counselor or someone you trust. You don't have to carry this alone.",
    ],
    "neutral": [
        "Got it! Is there anything on your mind you'd like to talk about — how are you actually feeling about it all?",
        "Thanks for sharing that. How are you holding up with everything going on?",
        "Noted. If there's anything weighing on you beyond the logistics, feel free to share — I'm here to listen.",
        "Okay! And how are you feeling about things overall? Sometimes it helps to just talk it out.",
        "Alright! Let me know if there's more you'd like to share — about how you're feeling, not just what's happening.",
    ],
    "unclear": [
        "I want to make sure I understand how you're feeling. Could you tell me a little more about what's on your mind?",
        "Hmm, I'm not quite sure how to read that. Can you share a bit more about what you're going through?",
        "I'd love to understand better — what's been happening lately that made you want to share that?",
        "It seems like there might be more behind what you said. Feel free to open up — I'm here without judgment.",
        "I want to be helpful, but I need a little more to go on. What's been on your mind lately?",
    ]
}

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm",
    "self-harm", "hurt myself", "don't want to live", "want to die",
    "no reason to live", "better off without me"
]

TOO_SHORT_THRESHOLD = 4  # words


def get_response(sentiment, user_text, confidence):
    lowered = user_text.lower()
    words = lowered.split()

    # Crisis check — always first
    if any(kw in lowered for kw in CRISIS_KEYWORDS):
        return (
            "I'm really concerned about what you just shared. Please know that you matter and you're not alone. "
            "It's important to reach out to someone who can help right now — a school counselor, a trusted friend or family member, "
            "or a mental health professional. You deserve support."
        ), "crisis"

    # Too short to classify well
    if len(words) <= TOO_SHORT_THRESHOLD and confidence < UNCLEAR_THRESHOLD:
        return random.choice(RESPONSES["unclear"]), "unclear"

    # Low confidence — unclear even for longer messages
    if confidence < UNCLEAR_THRESHOLD:
        return random.choice(RESPONSES["unclear"]), "unclear"

    return random.choice(RESPONSES.get(sentiment, RESPONSES["neutral"])), sentiment


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    cleaned = preprocess_text(user_message)
    sentiment = model.predict([cleaned])[0]

    proba = model.predict_proba([cleaned])[0]
    confidence = max(proba)
    confidence_pct = round(confidence * 100, 1)

    bot_reply, display_sentiment = get_response(sentiment, user_message, confidence)

    return jsonify({
        "sentiment": display_sentiment,
        "raw_sentiment": sentiment,
        "confidence": confidence_pct,
        "response": bot_reply
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)