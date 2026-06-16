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
    """Creates the user and messages tables in SQLite if they don't exist yet."""
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(username) REFERENCES users(username)
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


CONSECUTIVE_NEGATIVE_RESPONSES = [
    "I've noticed you've been feeling pretty stressed lately. That's completely valid — heavy feelings are real. How can I help lighten the load?",
    "You've shared a lot of tough moments today. I want you to know you're not alone in this. Even small breaks can help reset things. Have you taken time for yourself?",
    "It sounds like things have been really challenging. Remember that this feeling won't last forever, even though it feels heavy right now. What's one small thing that could help today?",
    "I hear that you're going through a difficult stretch. Please be gentle with yourself. Would it help to talk about what's been the hardest part?",
    "You've been carrying a lot. Stress like this deserves attention and care. Consider reaching out to someone you trust — a friend, counselor, or family member. You don't have to handle this alone.",
]


def get_response(sentiment, user_text, confidence, consecutive_negatives=0):
    lowered = user_text.lower()
    words = lowered.split()

    if any(kw in lowered for kw in CRISIS_KEYWORDS):
        return (
            "I'm really concerned about what you just shared. Please know that you matter and you're not alone. "
            "It's important to reach out to someone who can help right now — a school counselor, a trusted friend or family member, "
            "or a mental health professional. You deserve support."
        ), "crisis"

    # Check for 3+ consecutive negatives and provide supportive follow-up
    if consecutive_negatives >= 3 and sentiment == "negative":
        return random.choice(CONSECUTIVE_NEGATIVE_RESPONSES), "negative"

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
    error_message = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password")
        
        if not username or not email or not password:
            error_message = "Please fill out all fields."
        else:
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
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                cursor.execute("SELECT username, email FROM users WHERE username = ? OR email = ?", (username, email))
                existing = cursor.fetchone()
                conn.close()
                if existing:
                    if existing[0] == username:
                        error_message = "That username is already taken. Please choose another."
                    elif existing[1] == email:
                        error_message = "An account with that email already exists. Please log in or use a different email."
                    else:
                        error_message = "Username or email already exists."
                else:
                    error_message = "Unable to create account. Please try again."

    return render_template("signup.html", error=error_message)


@app.route("/login", methods=["GET", "POST"])
def login():
    error_message = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password")
        
        if not username or not password:
            error_message = "Please enter both username and password."
        else:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if not user:
                error_message = "No account found with that username. Please sign up first."
            elif not check_password_hash(user[3], password):
                error_message = "Password is incorrect. Please try again."
            else:
                session["username"] = user[1]  # Save username to session
                return redirect(url_for("home"))
            
    return render_template("login.html", error=error_message)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/get_history", methods=["GET"])
def get_history():
    """Retrieve chat history for the logged-in user."""
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_message, bot_response, sentiment, confidence, timestamp FROM messages WHERE username = ? ORDER BY id ASC",
            (session["username"],)
        )
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            confidence_value = row[4]
            if isinstance(confidence_value, (int, float)) and confidence_value <= 1:
                confidence_value = round(confidence_value * 100, 1)

            history.append({
                "id": row[0],
                "user_message": row[1],
                "bot_response": row[2],
                "sentiment": row[3],
                "confidence": confidence_value,
                "timestamp": row[5]
            })
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_mood_stats", methods=["GET"])
def get_mood_stats():
    """Get mood statistics for the logged-in user."""
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get all messages for this user ordered by timestamp
        cursor.execute(
            "SELECT sentiment, confidence FROM messages WHERE username = ? ORDER BY timestamp DESC LIMIT 100",
            (session["username"],)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({
                "total_messages": 0,
                "most_common_mood": None,
                "latest_mood": None,
                "consecutive_negatives": 0,
                "mood_distribution": {}
            })

        # Calculate statistics
        sentiments = [row[0] for row in rows]
        total = len(sentiments)
        latest = sentiments[0]
        
        # Count mood distribution
        mood_dist = {}
        for sentiment in sentiments:
            mood_dist[sentiment] = mood_dist.get(sentiment, 0) + 1
        
        most_common = max(mood_dist, key=mood_dist.get)
        
        # Detect consecutive negative messages
        consecutive_neg = 0
        for sentiment in sentiments:
            if sentiment == "negative":
                consecutive_neg += 1
            else:
                break
        
        return jsonify({
            "total_messages": total,
            "most_common_mood": most_common,
            "latest_mood": latest,
            "consecutive_negatives": consecutive_neg,
            "mood_distribution": mood_dist
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear chat history for the logged-in user."""
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE username = ?", (session["username"],))
        conn.commit()

        # Return fresh mood stats immediately after delete to avoid frontend timing/caching issues.
        cursor.execute(
            "SELECT sentiment, confidence FROM messages WHERE username = ? ORDER BY timestamp DESC LIMIT 100",
            (session["username"],)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({
                "success": True,
                "mood_stats": {
                    "total_messages": 0,
                    "most_common_mood": None,
                    "latest_mood": None,
                    "consecutive_negatives": 0,
                    "mood_distribution": {}
                }
            })

        sentiments = [row[0] for row in rows]
        total = len(sentiments)
        latest = sentiments[0]

        mood_dist = {}
        for sentiment in sentiments:
            mood_dist[sentiment] = mood_dist.get(sentiment, 0) + 1

        most_common = max(mood_dist, key=mood_dist.get)

        consecutive_neg = 0
        for sentiment in sentiments:
            if sentiment == "negative":
                consecutive_neg += 1
            else:
                break

        return jsonify({
            "success": True,
            "mood_stats": {
                "total_messages": total,
                "most_common_mood": most_common,
                "latest_mood": latest,
                "consecutive_negatives": consecutive_neg,
                "mood_distribution": mood_dist
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# Chat Routes
# -----------------------------------------------------------
@app.route("/")
def home():
    # If user is not logged in, force them to the login page
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


def save_message(username, user_msg, bot_msg, sentiment, confidence):
    """Save a chat message to the database."""
    try:
        if isinstance(confidence, (int, float)) and confidence <= 1:
            confidence = round(confidence * 100, 1)

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (username, user_message, bot_response, sentiment, confidence) VALUES (?, ?, ?, ?, ?)",
            (username, user_msg, bot_msg, sentiment, confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving message: {e}")


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

    # Get current consecutive negatives count
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sentiment FROM messages WHERE username = ? ORDER BY timestamp DESC LIMIT 5",
        (session["username"],)
    )
    recent = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    consecutive_neg = sum(1 for s in recent if s == "negative")
    
    bot_reply, display_sentiment = get_response(sentiment, user_message, confidence, consecutive_neg)

    # If not a crisis and we detected sarcasm in preprocessing, prefer a sarcasm-aware reply
    if display_sentiment != "crisis" and is_sarcasm:
        try:
            bot_reply = get_sarcasm_response(sarcasm_type)
            display_sentiment = "sarcasm"
        except Exception:
            # Fallback: leave original bot_reply
            pass

    # Save message to database as percent so history display is consistent.
    save_message(session["username"], user_message, bot_reply, display_sentiment, confidence_pct)

    return jsonify({
        "sentiment": display_sentiment,
        "raw_sentiment": sentiment,
        "confidence": confidence_pct,
        "response": bot_reply,
        "consecutive_negatives": consecutive_neg
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)