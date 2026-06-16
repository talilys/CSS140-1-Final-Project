import os
import re
import pickle
import random
import sqlite3
import requests
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from normalizer import normalize, get_sarcasm_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = "easeai_super_secret_key_change_this_later"

MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

UNCLEAR_THRESHOLD = 0.50
DATABASE_FILE = os.path.join(BASE_DIR, "data", "users.db")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


def init_db():
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


init_db()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Fallback responses (used if Ollama is offline) ──────────────────────────
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
    ],
}

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm",
    "self-harm", "hurt myself", "don't want to live", "want to die",
    "no reason to live", "better off without me"
]

TOO_SHORT_THRESHOLD = 4

CONSECUTIVE_NEGATIVE_RESPONSES = [
    "I've noticed you've been feeling pretty stressed lately. That's completely valid — heavy feelings are real. How can I help lighten the load?",
    "You've shared a lot of tough moments today. I want you to know you're not alone in this. Even small breaks can help reset things. Have you taken time for yourself?",
    "It sounds like things have been really challenging. Remember that this feeling won't last forever, even though it feels heavy right now. What's one small thing that could help today?",
    "I hear that you're going through a difficult stretch. Please be gentle with yourself. Would it help to talk about what's been the hardest part?",
    "You've been carrying a lot. Stress like this deserves attention and care. Consider reaching out to someone you trust — a friend, counselor, or family member. You don't have to handle this alone.",
]

VALID_SENTIMENTS = {"positive", "negative", "neutral", "unclear"}


# ── Ollama: mood detection + response in one call ───────────────────────────
def query_ollama(user_message, ml_sentiment, ml_confidence, consecutive_negatives=0):
    """
    Ask Ollama to do two things in one API call:
      1. Classify the sentiment (positive / negative / neutral / unclear)
      2. Estimate its own confidence (0–100)
      3. Write a short empathetic reply as EaseAI

    Returns: (ollama_sentiment, ollama_confidence, ollama_reply)
    All three are None if Ollama is unavailable.
    """

    system_prompt = (
        "You are EaseAI, a warm and empathetic student mental wellness companion for Filipino college students. "
        "You will receive a student's message. Your job is to do TWO things:\n\n"
        "1. Classify the student's emotional state as exactly one of: positive, negative, neutral, unclear.\n"
        "   - positive: doing well, happy, accomplished, relieved\n"
        "   - negative: stressed, anxious, overwhelmed, sad, burnt out\n"
        "   - neutral: just sharing info, no clear emotion\n"
        "   - unclear: too vague or ambiguous to classify\n\n"
        "2. Write a short empathetic reply (2–4 sentences max). "
        "Be conversational like a caring friend, not a textbook. "
        "Do NOT use bullet points or lists. "
        "Do NOT start your reply with 'I'. "
        "Never mention that you are an AI, Llama, or Ollama.\n\n"
        "You MUST respond in this exact JSON format and nothing else:\n"
        "{\n"
        '  "sentiment": "<positive|negative|neutral|unclear>",\n'
        '  "confidence": <integer between 0 and 100>,\n'
        '  "reply": "<your empathetic response here>"\n'
        "}"
    )

    consecutive_note = ""
    if consecutive_negatives >= 3:
        consecutive_note = (
            f" Note: this student has sent {consecutive_negatives} stressed messages in a row — "
            "be extra gentle and suggest they take a break or talk to someone."
        )

    user_prompt = (
        f"The ML sentiment model already classified this as '{ml_sentiment}' "
        f"with {ml_confidence:.0f}% confidence — use this as a hint but form your own judgment.{consecutive_note}\n\n"
        f"Student message: \"{user_message}\"\n\n"
        "Respond ONLY with the JSON object."
    )

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,   # lower = more consistent JSON output
                    "num_predict": 200,
                }
            },
            timeout=30
        )
        res.raise_for_status()
        raw = res.json().get("response", "").strip()

        # Strip markdown fences if Llama wraps the JSON anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        ollama_sentiment  = parsed.get("sentiment", "").lower().strip()
        ollama_confidence = float(parsed.get("confidence", 0))
        ollama_reply      = parsed.get("reply", "").strip()

        # Validate sentiment label
        if ollama_sentiment not in VALID_SENTIMENTS:
            print(f"[Ollama] Unrecognised sentiment '{ollama_sentiment}' — ignoring Ollama result.")
            return None, None, None

        # Clamp confidence to 0–100
        ollama_confidence = max(0.0, min(100.0, ollama_confidence))

        return ollama_sentiment, ollama_confidence, ollama_reply

    except requests.exceptions.ConnectionError:
        print("[Ollama] Not running — using ML model only.")
    except requests.exceptions.Timeout:
        print("[Ollama] Timed out — using ML model only.")
    except json.JSONDecodeError as e:
        print(f"[Ollama] JSON parse error: {e} — raw response was: {raw[:200]}")
    except Exception as e:
        print(f"[Ollama] Unexpected error: {e}")

    return None, None, None


# ── Highest-confidence wins ──────────────────────────────────────────────────
def resolve_sentiment(ml_sentiment, ml_confidence_pct,
                      ollama_sentiment, ollama_confidence):
    """
    Compare ML model vs Ollama confidence (both as 0–100).
    Whichever is more certain wins.
    Returns: (final_sentiment, final_confidence_pct, winner)
    winner is 'ml' or 'ollama' — useful for debugging.
    """
    if ollama_sentiment is None:
        return ml_sentiment, ml_confidence_pct, "ml"

    if ollama_confidence >= ml_confidence_pct:
        return ollama_sentiment, ollama_confidence, "ollama"
    else:
        return ml_sentiment, ml_confidence_pct, "ml"


def get_response(sentiment, user_text, confidence, consecutive_negatives=0):
    """Crisis and short-message guards run first, then Ollama reply is used."""
    lowered = user_text.lower()
    words   = lowered.split()

    if any(kw in lowered for kw in CRISIS_KEYWORDS):
        return (
            "I'm really concerned about what you just shared. Please know that you matter and you're not alone. "
            "It's important to reach out to someone who can help right now — a school counselor, a trusted friend or family member, "
            "or a mental health professional. You deserve support."
        ), "crisis"

    if len(words) <= TOO_SHORT_THRESHOLD and confidence < UNCLEAR_THRESHOLD * 100:
        return random.choice(RESPONSES["unclear"]), "unclear"

    if confidence < UNCLEAR_THRESHOLD * 100:
        return random.choice(RESPONSES["unclear"]), "unclear"

    return None, sentiment   # signal to caller: use Ollama reply


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error_message = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password")
        if not username or not email or not password:
            error_message = "Please fill out all fields."
        else:
            hashed_password = generate_password_hash(password)
            try:
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, hashed_password),
                )
                conn.commit()
                conn.close()
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT username, email FROM users WHERE username = ? OR email = ?",
                    (username, email),
                )
                existing = cursor.fetchone()
                conn.close()
                if existing:
                    error_message = (
                        "That username is already taken. Please choose another."
                        if existing[0] == username
                        else "An account with that email already exists."
                    )
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
                session["username"] = user[1]
                return redirect(url_for("home"))
    return render_template("login.html", error=error_message)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/get_history", methods=["GET"])
def get_history():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_message, bot_response, sentiment, confidence, timestamp "
            "FROM messages WHERE username = ? ORDER BY id ASC",
            (session["username"],),
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
                "timestamp": row[5],
            })
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_mood_stats", methods=["GET"])
def get_mood_stats():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sentiment, confidence FROM messages WHERE username = ? "
            "ORDER BY timestamp DESC LIMIT 100",
            (session["username"],),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return jsonify({
                "total_messages": 0, "most_common_mood": None,
                "latest_mood": None, "consecutive_negatives": 0,
                "mood_distribution": {},
            })
        sentiments = [row[0] for row in rows]
        mood_dist  = {}
        for s in sentiments:
            mood_dist[s] = mood_dist.get(s, 0) + 1
        consecutive_neg = 0
        for s in sentiments:
            if s == "negative":
                consecutive_neg += 1
            else:
                break
        return jsonify({
            "total_messages": len(sentiments),
            "most_common_mood": max(mood_dist, key=mood_dist.get),
            "latest_mood": sentiments[0],
            "consecutive_negatives": consecutive_neg,
            "mood_distribution": mood_dist,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE username = ?", (session["username"],))
        conn.commit()
        conn.close()
        return jsonify({
            "success": True,
            "mood_stats": {
                "total_messages": 0, "most_common_mood": None,
                "latest_mood": None, "consecutive_negatives": 0,
                "mood_distribution": {},
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


def save_message(username, user_msg, bot_msg, sentiment, confidence):
    try:
        if isinstance(confidence, (int, float)) and confidence <= 1:
            confidence = round(confidence * 100, 1)
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (username, user_message, bot_response, sentiment, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, user_msg, bot_msg, sentiment, confidence),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving message: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data         = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    # ── Step 1: Normalise (emojis, slangs, sarcasm detection) ───────────────
    enriched, is_sarcasm, sarcasm_type = normalize(user_message)
    cleaned = preprocess_text(enriched)

    # ── Step 2: ML model classification ─────────────────────────────────────
    ml_sentiment     = model.predict([cleaned])[0]
    proba            = model.predict_proba([cleaned])[0]
    ml_confidence    = max(proba)
    ml_confidence_pct = round(ml_confidence * 100, 1)

    # ── Step 3: Fetch recent history for consecutive-negative check ──────────
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sentiment FROM messages WHERE username = ? ORDER BY timestamp DESC LIMIT 5",
        (session["username"],),
    )
    recent = [row[0] for row in cursor.fetchall()]
    conn.close()
    consecutive_neg = sum(1 for s in recent if s == "negative")

    # ── Step 4: Crisis / short-message guards ────────────────────────────────
    guard_reply, guard_sentiment = get_response(
        ml_sentiment, user_message, ml_confidence_pct, consecutive_neg
    )

    if guard_reply is not None:
        # Crisis or unclear — skip Ollama entirely
        save_message(session["username"], user_message, guard_reply, guard_sentiment, ml_confidence_pct)
        return jsonify({
            "sentiment": guard_sentiment,
            "raw_sentiment": ml_sentiment,
            "confidence": ml_confidence_pct,
            "response": guard_reply,
            "consecutive_negatives": consecutive_neg,
        })

    # ── Step 5: Query Ollama (mood + confidence + reply in one call) ─────────
    ollama_sentiment, ollama_confidence, ollama_reply = query_ollama(
        user_message, ml_sentiment, ml_confidence_pct, consecutive_neg
    )

    # ── Step 6: Highest confidence wins ─────────────────────────────────────
    final_sentiment, final_confidence, winner = resolve_sentiment(
        ml_sentiment, ml_confidence_pct,
        ollama_sentiment, ollama_confidence
    )

    print(f"[Sentiment] ML={ml_sentiment}@{ml_confidence_pct}% | "
          f"Ollama={ollama_sentiment}@{ollama_confidence}% | "
          f"Winner={winner} → {final_sentiment}@{final_confidence}%")

    # ── Step 7: Pick the reply ───────────────────────────────────────────────
    if ollama_reply:
        bot_reply = ollama_reply
    else:
        bot_reply = random.choice(RESPONSES.get(final_sentiment, RESPONSES["neutral"]))

    # ── Step 8: Sarcasm override (always takes priority over Ollama reply) ───
    display_sentiment = final_sentiment
    if is_sarcasm:
        try:
            bot_reply         = get_sarcasm_response(sarcasm_type)
            display_sentiment = "sarcasm"
        except Exception:
            pass

    save_message(session["username"], user_message, bot_reply, display_sentiment, final_confidence)

    return jsonify({
        "sentiment":             display_sentiment,
        "raw_sentiment":         ml_sentiment,
        "confidence":            final_confidence,
        "response":              bot_reply,
        "consecutive_negatives": consecutive_neg,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)