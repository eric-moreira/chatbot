import nltk
import random
import string
from nltk.corpus import wordnet
from flask import Flask, jsonify, request

# Define the pairs of inputs and responses
pairs = {
    "hi": ["Hello!", "Hi there!"],
    "how are you?": ["I'm doing well, thank you.", "I'm good. How are you?"],
    "what is your name?": ["My name is Eric-AI.", "You can call me Eric-AI."],
    "bye": ["Goodbye!", "See you later."],
    "default": ["I'm sorry, I don't understand.", "Can you please rephrase that?", "I'm not sure I understand."],
}

# Preprocessing the data
nltk.download('punkt')
lemmer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    text = text.lower().translate(remove_punct_dict)
    words = nltk.word_tokenize(text)
    words = [lemmer.lemmatize(word) for word in words]
    return words

# Transforming the data
def response(user_response):
    words = preprocess(user_response)
    response_text = ""
    matched = False
    for word in words:
        for key in pairs.keys():
            if wordnet.synsets(word) and word in key:
                response_text += random.choice(pairs[key]) + " "
                matched = True
                break
        if matched:
            break
    if not matched:
        response_text = random.choice(pairs["default"])
    return response_text

# Initialize Flask app
app = Flask(__name__)

# Define route for receiving POST requests
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_response = data['message']
    response_text = response(user_response)
    return jsonify({'message': response_text})

# Start the app
if __name__ == '__main__':
    app.run()
