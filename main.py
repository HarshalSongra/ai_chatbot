import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
from tensorflow.python.framework import ops
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
        doc_x.append(wrds)
        doc_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])


# arranging data
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# saperating data for training and output
for x , doc in enumerate(doc_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(doc_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)  


training = np.array(training)
output = np.array(output)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
