from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained BERT model state dictionary
model.load_state_dict(torch.load('./trained/bert_model.pkl', map_location=torch.device('cpu')))
model.eval()

# Function to preprocess text
def preprocess_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return inputs

# Classify text
def classify_text(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    return probabilities

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data['text']

    print("Received text:", text)  # Debug statement

    # Perform text classification
    probabilities = classify_text(text)

    # Get probabilities for human and ChatGPT
    human_probability = probabilities[0][0].item()
    chatgpt_probability = probabilities[0][1].item()

    print("Probabilities (Raw):", probabilities)  # Debug statement
    print("Human probability:", human_probability)  # Debug statement
    print("ChatGPT probability:", chatgpt_probability)  # Debug statement

    # Return the probabilities in JSON response
    return jsonify({
        'human_probability': human_probability,
        'chatgpt_probability': chatgpt_probability
    })

if __name__ == '__main__':
    app.run(debug=True)
