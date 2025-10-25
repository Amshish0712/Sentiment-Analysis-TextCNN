import os
import pickle
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MAX_LEN = 50

def clean_text(text: str):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower().strip()

def load_tokenizer(tokenizer_path: Path):
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)

def load_keras_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)

def preprocess_texts_for_keras(tokenizer, texts, max_len=MAX_LEN):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return seqs

def keras_predict(model, tokenizer, text, max_len=MAX_LEN):
    text = clean_text(text)
    x = preprocess_texts_for_keras(tokenizer, [text], max_len=max_len)
    probs = model.predict(x)[0]

    if np.ndim(probs) == 0:
        score = float(probs)
        label = "positive" if score >= 0.5 else "negative"
        return {"score": score, "label": label}

    probs = probs.tolist()
    pos = float(probs[-1]) if len(probs) >= 1 else 0.5
    label = "positive" if pos >= 0.5 else "negative"
    return {"score": pos, "label": label}

AVAILABLE_MODELS = {}
TOKENIZER = None
DEFAULT_MODEL = None

def discover_and_load():
    global AVAILABLE_MODELS, TOKENIZER, DEFAULT_MODEL
    tokenizer_path = MODELS_DIR / "tokenizer.pkl"
    if tokenizer_path.exists():
        TOKENIZER = load_tokenizer(tokenizer_path)
        print("✅ Tokenizer loaded.")
    else:
        print("⚠️ tokenizer.pkl not found.")

    for file in MODELS_DIR.iterdir():
        if file.suffix in {".h5", ".hdf5"}:
            try:
                model = load_keras_model(file)
                AVAILABLE_MODELS[file.stem.lower()] = {"model": model}
                print(f"✅ Loaded model: {file.name}")
            except Exception as e:
                print(f"❌ Failed to load {file.name}: {e}")

    if "textcnn" in AVAILABLE_MODELS:
        DEFAULT_MODEL = "textcnn"
        print("🌟 Default model: TextCNN")
    elif AVAILABLE_MODELS:
        DEFAULT_MODEL = next(iter(AVAILABLE_MODELS.keys()))
        print(f"⚠️ Using fallback model: {DEFAULT_MODEL}")
    else:
        print("❌ No models found.")

discover_and_load()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if not AVAILABLE_MODELS or not DEFAULT_MODEL:
        return jsonify({"error": "Model not loaded."}), 500

    model = AVAILABLE_MODELS[DEFAULT_MODEL]["model"]
    try:
        result = keras_predict(model, TOKENIZER, text, max_len=MAX_LEN)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
