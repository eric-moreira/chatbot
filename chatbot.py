import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the pairs of inputs and responses
pairs = [
    ["hi", ["Hello!", "Hi there!"]],
    ["how are you?", ["I'm doing well, thank you.", "I'm good. How are you?"]],
    ["what is your name?", ["My name is Eric-AI.", "You can call me Eric-AI."]],
    ["bye", ["Goodbye!", "See you later."]],
]

# Preprocessing the data
nltk.download('punkt')
nltk.download('wordnet')
lemmer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    text = text.lower().translate(remove_punct_dict)
    words = nltk.word_tokenize(text)
    words = [lemmer.lemmatize(word) for word in words]
    return " ".join(words)

# Transforming the data
def response(user_response):
    pairs.append(['You', [user_response]])
    TfidfVec = TfidfVectorizer(tokenizer=preprocess, stop_words='english')
    tfidf = TfidfVec.fit_transform([pair[0] for pair in pairs])
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    if score == 0:
        return "I'm sorry, I don't understand."
    else:
        return pairs[idx][1][random.choice(range(len(pairs[idx][1])))]

# Define a function to generate responses based on user input
def chatbot():
    print("Hi! I'm AI. How can I help you today?")
    while True:
        user_response = input()
        if user_response.lower() == 'bye':
            print("Goodbye!")
            break
        else:
            print(response(user_response))

# Call the chatbot function to start the conversation
chatbot()
