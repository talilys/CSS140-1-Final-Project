"""
app.py
EaseAI - Student Stress and Mood Detection Chatbot with Authentication
Improved: confidence-based unclear detection, richer responses,
context-aware follow-ups, better crisis handling, and SQLite-based auth.
"""

import os
import re
import pickle
import random
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from normalizer import normalize, get_sarcasm_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# CRITICAL: Set a secret key to enable session tracking (login states)
app.secret_key = "easeai_super_secret_key_change_this_later"

# Load model relative to the app file so the server works even when started from another CWD.
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Confidence threshold below which we ask for more context
UNCLEAR_THRESHOLD = 0.50
DATABASE_FILE = os.path.join(BASE_DIR, "data", "users.db")


# -----------------------------------------------------------
# Database Helper Functions
# -----------------------------------------------------------
def init_db():
    """Creates the user table in SQLite if it doesn't exist yet."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database immediately on app start
init_db()


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

    if any(kw in lowered for kw in CRISIS_KEYWORDS):
        return (
            "I'm really concerned about what you just shared. Please know that you matter and you're not alone. "
            "It's important to reach out to someone who can help right now — a school counselor, a trusted friend or family member, "
            "or a mental health professional. You deserve support."
        ), "crisis"

    if len(words) <= TOO_SHORT_THRESHOLD and confidence < UNCLEAR_THRESHOLD:
        return random.choice(RESPONSES["unclear"]), "unclear"

    if confidence < UNCLEAR_THRESHOLD:
        return random.choice(RESPONSES["unclear"]), "unclear"

    return random.choice(RESPONSES.get(sentiment, RESPONSES["neutral"])), sentiment


# -----------------------------------------------------------
# Authentication Routes
# -----------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username").strip()
        email = request.form.get("email").strip()
        password = request.form.get("password")
        
        if not username or not email or not password:
            return "Please fill out all fields.", 400
            
        # Securely hash the password before saving
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                           (username, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "Username or Email already exists.", 400
            
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        # user[3] corresponds to the hashed password column
        if user and check_password_hash(user[3], password):
            session["username"] = user[1] # Save username to session
            return redirect(url_for("home"))
        else:
            return "Invalid username or password.", 401
            
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


# -----------------------------------------------------------
# Chat Routes
# -----------------------------------------------------------
@app.route("/")
def home():
    # If user is not logged in, force them to the login page
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    enriched, is_sarcasm, sarcasm_type = normalize(user_message)
    cleaned = preprocess_text(enriched)
    sentiment = model.predict([cleaned])[0]

    proba = model.predict_proba([cleaned])[0]
    confidence = max(proba)
    confidence_pct = round(confidence * 100, 1)

    bot_reply, display_sentiment = get_response(sentiment, user_message, confidence)

    # If not a crisis and we detected sarcasm in preprocessing, prefer a sarcasm-aware reply
    if display_sentiment != "crisis" and is_sarcasm:
        try:
            bot_reply = get_sarcasm_response(sarcasm_type)
            display_sentiment = "sarcasm"
        except Exception:
            # Fallback: leave original bot_reply
            pass

    return jsonify({
        "sentiment": display_sentiment,
        "raw_sentiment": sentiment,
        "confidence": confidence_pct,
        "response": bot_reply
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)