# EaseAI: Student Stress and Mood Detection Chatbot

A sentiment analysis chatbot built with **TF-IDF + Logistic Regression** (scikit-learn)
and served through a **Flask** web app. EaseAI classifies student messages as
**positive**, **negative**, or **neutral**, and responds with supportive,
mood-appropriate messages.

## Project Structure

```
EaseAI/
├── app.py                 # Flask web application
├── train_model.py         # Trains and saves the ML model
├── requirements.txt
├── data/
│   └── dataset.csv        # Labeled training data
├── model/
│   └── sentiment_model.pkl  # Saved TF-IDF + Logistic Regression pipeline
├── templates/
│   └── index.html         # Chat UI
└── static/
    ├── style.css
    └── script.js
```

## Setup Instructions (VS Code)

1. **Open the project folder** in VS Code (`File > Open Folder` → select `EaseAI`).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (this creates `model/sentiment_model.pkl`):
   ```bash
   python train_model.py
   ```
   You'll see accuracy, classification report, and sample predictions printed
   in the terminal — useful for your paper's results section.

5. **Run the Flask app**:
   ```bash
   python app.py
   ```

6. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## How It Works

1. **Text Input** – User types a message in the chat UI.
2. **Preprocessing** – Text is lowercased, punctuation/numbers removed, extra
   whitespace stripped (`preprocess_text()` in both `train_model.py` and `app.py`).
3. **TF-IDF Vectorization** – Converts cleaned text into numerical features
   (unigrams + bigrams, max 3000 features, English stop words removed).
4. **Logistic Regression** – Predicts sentiment: `positive`, `negative`, or `neutral`.
5. **Response Generation** – Based on the predicted sentiment, EaseAI selects a
   supportive response from a curated bank. A simple keyword-based safety check
   flags crisis-related language and responds with a supportive referral message.

## Improving the Model

- Add more rows to `data/dataset.csv` (more examples = better accuracy).
- Re-run `python train_model.py` after every dataset change to retrain and
  overwrite `model/sentiment_model.pkl`.
- You can tune `TfidfVectorizer` parameters (e.g., `max_features`, `ngram_range`)
  or `LogisticRegression` parameters (e.g., `C`, `class_weight`) in `train_model.py`.

## Disclaimer

EaseAI is an academic mini-project and is **not** a substitute for professional
mental health services. The crisis-keyword response in `app.py` encourages users
to seek help from a counselor or trusted person — it is not a clinical safety system.
