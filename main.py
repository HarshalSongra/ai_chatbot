import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open('intents.json') as file:
    data =  json.load(file)

# have each word in pattern
words = []
# tags
labels = []
# patterns
doc_x = []
doc_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        # saperates the word from string
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        doc_x.append(pattern)
        doc_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

