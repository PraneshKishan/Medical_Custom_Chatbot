import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize the lemmatizer and load the necessary files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'D:\Medical Chatbot\intents.json').read()) #Enter your correct intents.json file Path
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Creates a bag of words array for the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predicts the class (intent) for the given sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Fetches a random response for the predicted intent."""
    if not intents_list:
        return "Sorry, I didn't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    
    return "Sorry, I didn't find a suitable response."

print("GO! Bot is running!")

while True:
    try:
        message = input("You: ")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("Bot:", res)
    except KeyboardInterrupt:
        print("\nExiting... Have a great day!")
        break
