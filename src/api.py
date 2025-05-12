from flask import Flask, request, render_template_string, jsonify
import torch
from transformers import AutoTokenizer
from model import MentalHealthClassifier
from rag import AdviceRAG

app = Flask(__name__)

# Config
MODEL_NAME = "distilbert-base-uncased"
LABELS = ['Anxiety', 'Bi-Polar', 'Depression', 'Normal', 'Personality Disorder', 'Stress', 'Suicidal']
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MentalHealthClassifier(MODEL_NAME, num_labels=len(LABELS))
model.load_state_dict(torch.load("models/model_v1.pt", map_location=device))
model.eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load RAG
rag = AdviceRAG("data/advice/mental_health_advice.txt")

# Emoji map
emojis = {
    "Depression": "😔",
    "Bi-Polar": "🔄",
    "Normal": "🙂",
    "Suicidal": "🆘",
    "Anxiety": "😰",
    "Stress": "😫",
    "Personality Disorder": "🧠"
}


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()
    return id2label[prediction]


@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!doctype html>
    <html lang="en">
    <head>
        <title>Mental Health Support</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(to right, #74ebd5, #ACB6E5);
                margin: 0; padding: 0;
                display: flex; justify-content: center; align-items: center;
                height: 100vh;
            }
            .container {
                background-color: white;
                border-radius: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                padding: 40px;
                max-width: 600px;
                width: 90%;
                text-align: center;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 10px;
                border: 2px solid #ccc;
                border-radius: 8px;
                font-size: 16px;
                resize: none;
            }
            button {
                margin-top: 20px;
                padding: 10px 25px;
                background-color: #5C6BC0;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background-color: #3F51B5;
            }
            .mic {
                font-size: 24px;
                cursor: pointer;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Mental Health Sentiment Classifier</h2>
            <form method="POST" action="/predict_form">
                <textarea name="text" id="inputText" placeholder="How are you feeling today?" required></textarea><br>
                <button type="submit">Analyze</button>
                <span class="mic" onclick="startListening()">🎤</span>
            </form>
        </div>

        <audio autoplay loop>
            <source src="https://www.bensound.com/bensound-music/bensound-tomorrow.mp3" type="audio/mpeg">
        </audio>

        <script>
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById("inputText").value = transcript;
            };
            function startListening() {
                recognition.start();
            }
        </script>
    </body>
    </html>
    """)


@app.route("/predict_form", methods=["POST"])
def predict_form():
    text = request.form.get("text", "")
    if not text:
        return "No input provided.", 400

    sentiment = predict_sentiment(text)
    advice = rag.retrieve(text,sentiment)
    emoji = emojis.get(sentiment, "🧠")

    return render_template_string(f"""
    <!doctype html>
    <html lang="en">
    <head>
        <title>Result</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(to right, #ffecd2, #fcb69f);
                margin: 0; padding: 0;
                display: flex; justify-content: center; align-items: center;
                height: 100vh;
            }}
            .result {{
                background: white;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                padding: 40px;
                text-align: center;
                max-width: 600px;
                width: 90%;
            }}
            .emoji {{
                font-size: 64px;
                animation: bounce 2s infinite;
            }}
            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-10px); }}
            }}
            p {{ font-size: 18px; color: #333; }}
            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #f06292;
                color: white;
                border-radius: 8px;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="result">
            <div class="emoji">{emoji}</div>
            <h2>Your Mental Health Result</h2>
            <p><strong>Sentiment:</strong> {sentiment}</p>
            <p><strong>Advice:</strong> {advice}</p>
            <a href="/">← Try again</a>
        </div>
    </body>
    </html>
    """)


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment = predict_sentiment(text)
    advice = rag.retrieve(text,sentiment)
    return jsonify({"sentiment": sentiment, "advice": advice})


if __name__ == "__main__":
    app.run(debug=True)