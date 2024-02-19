from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import threading
app = Flask(__name__)

# Load model and tokenizer
# model_name = "qanastek/51-languages-classifier"
model_name = "/app/models/51-languages-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Create a lock object
lock = threading.Lock()

@app.route('/classify', methods=['POST'])
def classify_text():
    if lock.locked():
        return jsonify({'error': 'Classifier is busy'}), 429  # 429 Too Many Requests

    with lock:
        text = request.json['text']
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = classifier(text)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)